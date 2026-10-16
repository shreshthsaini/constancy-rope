"""Verify partial caches, then publish the final all-image scoring tasks."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

NOTES = Path(__file__).resolve().parent
PROJECT = NOTES.parent.parent
sys.path.insert(0, str(PROJECT))
from crope.metrics.io import digest, discover_samples, file_signature, read_jsonl, write_json


def main():
    scratch = Path("/scratch/09032/saini_2/constancy_rope")
    queue = scratch / "taskq"
    root = scratch / "eval_krea"
    target = root / "main2"
    generation = json.loads((NOTES / "main2_generation_check.json").read_text())
    assert generation["complete"] and generation["checked_images"] == 188
    assert len(list((queue / "done").glob("53_c*_main2_sega_ctm_eta10.sh"))) == 27
    assert (queue / "done/62_00_gpu_prepass_main2.sh").is_file()
    assert not (target / "per_image.csv").exists()
    records = discover_samples(scratch / "out_krea/main2", read_jsonl(scratch / "prompts/aesthetic4k_eval.jsonl"), "4096x4096")
    assert len(records) == 188
    by_path = {row["path"]: row for row in records}
    baseline = json.loads((root / "main/metrics.json").read_text())
    cache_path = target / "evaluation_cache.json"
    merged = json.loads(cache_path.read_text())
    assert len(merged) == 195
    receipts = []
    for name, metrics in (("main2_cpu_prepass", ("luminance_octaves", "tiling")),
                          ("main2_gpu_prepass", ("clip", "pickscore", "musiq", "clipiqa"))):
        report = json.loads((root / name / "metrics.json").read_text())
        cache = json.loads((root / name / "evaluation_cache.json").read_text())
        assert report["complete"] and not report["errors"], name
        assert report["versions"] == baseline["versions"], name
        assert report["model_revisions"] == baseline["model_revisions"], name
        for setting in ("crop_seed", "crops", "res"):
            assert report["config"][setting] == baseline["config"][setting], (name, setting)
        settings = dict(schema=report["metric_schema"], crop_seed=report["config"]["crop_seed"],
                        crops=report["config"]["crops"], versions=report["versions"],
                        model_revisions=report["model_revisions"])
        count = 0
        seen = set()
        for row in report["per_image"]:
            assert row["path"] in by_path and row["path"] not in seen
            seen.add(row["path"])
            current = by_path[row["path"]]
            assert row["config_hash"] == current["config_hash"]
            for metric in metrics:
                key = digest(dict(file=file_signature(current["path"]), prompt=current.get("prompt"),
                                  prompt_id=current["prompt_id"], metric=metric, settings=settings,
                                  source_config_hash=current.get("config_hash"), extra=None))
                assert key in cache and key not in merged, (name, metric)
                value = cache[key]
                if name == "main2_gpu_prepass":
                    assert value["provenance"] == baseline["status"][metric]["provenance"]
                    assert all(math.isfinite(score) for score in value["scores"].values())
                merged[key] = value
                count += 1
        receipts.append(dict(source=name, images=len(seen), imported_entries=count))
    ir_source = root / "ir/main2_prepass.csv"
    ir_target = root / "ir/main2.csv"
    assert not ir_target.exists()
    with ir_source.open() as stream:
        rewards = list(csv.DictReader(stream))
    assert len({row["png"] for row in rewards}) == len(rewards)
    for row in rewards:
        current = by_path[row["png"]]
        assert all(row[key] == current[key] for key in ("prompt_id", "method", "res"))
        assert math.isfinite(float(row["imagereward"]))
    expected_tasks = json.loads((NOTES / "main2_eval_preparation.json").read_text())["score_tasks"]
    for task in expected_tasks:
        path = queue / "staging" / task["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == task["sha256"]
        assert not (queue / "pending" / task["name"]).exists()
        subprocess.run(["bash", "-n", str(path)], check=True)
    backup = target / "evaluation_cache.reference_only.json"
    assert not backup.exists()
    shutil.copy2(cache_path, backup)
    write_json(cache_path, merged)
    shutil.copy2(ir_source, ir_target)
    for task in expected_tasks:
        (queue / "staging" / task["name"]).rename(queue / "pending" / task["name"])
    receipt = dict(at=datetime.now().astimezone().isoformat(), sources=receipts,
                   total_cache_entries=len(merged), imagereward_imported=len(rewards),
                   final_tasks_published=[task["name"] for task in expected_tasks],
                   checks="Current image signatures/config hashes, scorer settings/revisions/versions/provenance, finite learned scores, reward path identities, task hashes")
    (NOTES / "main2_cache_merge.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
