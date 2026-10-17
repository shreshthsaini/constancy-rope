import csv
import hashlib
import json

import pytest

from scripts.dev3_analysis import METRICS, VARIANTS, analyze, decision


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def campaign(tmp_path, reference="pilot"):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(dict(id=str(i), prompt=f"prompt {i}")) + "\n" for i in range(10)))
    for method in ("sega_official", *VARIANTS):
        root = reference if method == "sega_official" else f"dev3_{method}"
        scores, ir = [], []
        for i in range(8):
            png = tmp_path / "out_krea" / root / method / "4096x4096" / f"{i}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            png.write_bytes(f"fixture image {i}".encode())
            config = dict(method=method, prompt_id=str(i), prompt=f"prompt {i}", seed=0,
                          derived_seed=i, steps=28, guidance=4.5, res="4096x4096", grid=[256, 256],
                          rope={"method": "sega_official"}, eta=1., profile_sha256="profile",
                          temperature="none", model="flux", model_revision="revision", dtype="bfloat16",
                          batch_size=1, torch="torch", diffusers="diffusers", implementation_sha256="source")
            sigmas = [1 - s / 28 for s in range(28)]
            record = dict(config=config, complete=True, image=str(png), sigmas=sigmas,
                          gains=[dict(sigma=s) for s in sigmas], png_sha256=hashlib.sha256(png.read_bytes()).hexdigest())
            png.with_suffix(".json").write_text(json.dumps(record))
            common = dict(method=method, prompt_id=str(i), res="4096x4096")
            value = 1. if method == "sega_official" else 2.
            scores.append({**common, "seed": 0, "path": str(png), **{m: value for m in METRICS[1:]}})
            ir.append({**common, "png": str(png), "imagereward": value})
        table = tmp_path / "eval_krea" / root / ("4096x4096/per_image.csv" if root == "pilot" else "per_image.csv")
        write_csv(table, scores)
        write_csv(tmp_path / "eval_krea/ir" / f"{root}.csv", ir)
    return tmp_path / "eval_krea", prompts, tmp_path / "report"


@pytest.mark.parametrize("reference", ["pilot", "dev3_sega_official"])
def test_full_analysis_excludes_heldout_and_records_exploratory_cap(tmp_path, reference):
    args = campaign(tmp_path, reference)
    report = analyze(*args, reference_root=reference)
    assert report["prompt_ids"] == [str(i) for i in range(8)]
    assert report["selected"] == ["sega_ctm_lag", "sega_rms_normalized"]
    assert not report["automatic_heldout_authorized"]
    assert all(m in report["rows"][0] for m in METRICS)


@pytest.mark.parametrize("problem", ["method", "eta", "sigma", "gain_sigma", "seed", "provenance", "checksum", "incomplete"])
def test_sidecar_provenance_failures(tmp_path, problem):
    args = campaign(tmp_path)
    path = tmp_path / "out_krea/dev3_sega_ctm_lag/sega_ctm_lag/4096x4096/0.json"
    record = json.loads(path.read_text())
    if problem in ("method", "eta", "seed"):
        record["config"][problem] = {"method": "sega_ctm", "eta": .5, "seed": 1}[problem]
    elif problem == "sigma":
        record["sigmas"][5] -= .001
        record["gains"][5]["sigma"] -= .001
    elif problem == "gain_sigma":
        record["gains"][5]["sigma"] -= .001
    elif problem == "provenance":
        record["config"]["model_revision"] = "different"
    elif problem == "checksum":
        record["png_sha256"] = "wrong"
    else:
        record["complete"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        analyze(*args)


@pytest.mark.parametrize("problem", ["path", "duplicate", "heldout_id", "wrong_method", "nan", "duplicate_path"])
def test_score_identity_and_metric_failures(tmp_path, problem):
    args = campaign(tmp_path)
    table = args[0] / "dev3_sega_rms_uniform/per_image.csv"
    with table.open() as stream:
        rows = list(csv.DictReader(stream))
    if problem == "path":
        rows[0]["path"] = rows[0]["path"].replace("dev3_sega_rms_uniform", "elsewhere")
    elif problem == "duplicate":
        rows.append(rows[0].copy())
    elif problem == "heldout_id":
        rows[0]["prompt_id"] = "8"
    elif problem == "wrong_method":
        rows[0]["method"] = "sega_official"
    elif problem == "nan":
        rows[0]["pickscore"] = "nan"
    else:
        rows[0]["path"] = rows[1]["path"]
    write_csv(table, rows)
    with pytest.raises(ValueError):
        analyze(*args)


def test_strict_gate_ties_clip_boundary_and_ranking():
    means = {"sega_official": {m: 0. for m in METRICS}}
    for name, ir, musiq, clip in [("ir_tie", 0, 1, 0), ("musiq_tie", 1, 0, 0),
                                 ("clip_bad", 5, 2, -.30001), ("boundary", 2, 2, -.3),
                                 ("best", 3, 1, 0), ("third", 1, 5, 0)]:
        means[name] = dict(imagereward=ir, musiq=musiq, clip=clip, pickscore=0., musiq_crop1024=0.)
    rows, selected = decision(means)
    assert selected == ["best", "boundary"]
    assert {r["variant"] for r in rows if r["eligible"]} == {"best", "boundary", "third"}
