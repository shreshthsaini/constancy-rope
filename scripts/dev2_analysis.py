"""Apply the registered second-campaign dev rule to exactly eight matched prompts."""

import argparse
import csv
import json
import math
from pathlib import Path


VARIANTS = {"alias_temp": "alias_temp", "alias_temp_sref": "alias_temp_sref",
            "sega_ctm_eta05": "sega_ctm", "sega_ctm_eta10": "sega_ctm"}
METRICS = ("imagereward", "musiq", "clip")


def read_scores(path, method, ids, metrics):
    with Path(path).open() as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["method"] == method and row["res"] == "4096x4096"]
    if len(rows) != len(ids) or {row["prompt_id"] for row in rows} != set(ids):
        raise ValueError(f"{path}: require exactly the eight dev IDs once for {method}")
    result = {}
    for row in rows:
        if "seed" in row and int(row["seed"]) != 0:
            raise ValueError(f"{path}: expected seed 0")
        values = {metric: float(row[metric]) for metric in metrics}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{path}: missing or nonfinite required metric")
        values["source_path"] = str(Path(row.get("path", row.get("png", ""))).resolve())
        result[row["prompt_id"]] = values
    return result


def validate_pair(scores, reward, variant, method, prompt_id):
    path = Path(scores["source_path"])
    if scores["source_path"] != reward["source_path"]:
        raise ValueError("ImageReward and evaluation must score the same image path")
    root_name = "pilot" if variant == "sega_official" else f"dev2_{variant}"
    if (not path.is_file() or path.parent.name != "4096x4096" or
            path.parent.parent.name != method or path.parent.parent.parent.name != root_name):
        raise ValueError("Score image is outside the expected variant root")
    config = json.loads(path.with_suffix(".json").read_text())["config"]
    expected = dict(seed=0, steps=28, guidance=4.5, res="4096x4096", method=method, prompt_id=prompt_id)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Scored image sidecar differs from the registered dev protocol")
    if config["rope"]["method"] != "sega_official":
        raise ValueError("Scored image uses the wrong RoPE base")
    expected_eta = {"sega_ctm_eta05": 0.5, "sega_ctm_eta10": 1.0}.get(variant)
    if expected_eta is not None and config["eta"] != expected_eta:
        raise ValueError("Scored image uses the wrong sega_ctm eta")
    if variant != "sega_official" and config["temperature"] != "none":
        raise ValueError("New candidates must have no extra temperature")


def decision(means):
    reference = means["sega_official"]
    rows = []
    for variant, values in means.items():
        delta = {metric: values[metric] - reference[metric] for metric in METRICS}
        eligible = (variant != "sega_official" and delta["imagereward"] > 0
                    and delta["musiq"] > 0 and delta["clip"] >= -0.3)
        rows.append({"variant": variant, "n": 8, **values,
                     **{f"delta_{key}": value for key, value in delta.items()}, "eligible": eligible})
    eligible = sorted((row for row in rows if row["eligible"]),
                      key=lambda row: (-row["delta_imagereward"], -row["delta_musiq"], row["variant"]))
    selected = [row["variant"] for row in eligible[:2]]
    for row in rows:
        row["selected"] = row["variant"] in selected
    return rows, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    ids = [json.loads(line)["id"] for line in args.prompts.read_text().splitlines()][:8]
    assert len(ids) == len(set(ids)) == 8
    means, paired = {}, []
    for variant, method in {"sega_official": "sega_official", **VARIANTS}.items():
        name = "pilot" if variant == "sega_official" else f"dev2_{variant}"
        table = args.eval_root / name / "per_image.csv"
        if variant == "sega_official":
            table = args.eval_root / "pilot/4096x4096/per_image.csv"
        scores = read_scores(table, method, ids, ("musiq", "clip"))
        reward = read_scores(args.eval_root / "ir" / f"{name}.csv", method, ids, ("imagereward",))
        for pid in ids:
            validate_pair(scores[pid], reward[pid], variant, method, pid)
        records = [{"variant": variant, "prompt_id": pid, **scores[pid], **reward[pid]} for pid in ids]
        paired.extend(records)
        means[variant] = {metric: sum(row[metric] for row in records) / 8 for metric in METRICS}
    rows, selected = decision(means)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in (("dev_table.csv", rows), ("dev_per_prompt.csv", paired)):
        with (args.out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    report = {"purpose": "Dev selection only; not held-out results", "prompt_ids": ids,
              "rule": "IR delta > 0 AND full-res MUSIQ delta > 0 AND CLIP delta >= -0.3",
              "ranking": "IR delta descending, MUSIQ delta descending, variant name ascending",
              "selected": selected, "rows": rows}
    (args.out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
