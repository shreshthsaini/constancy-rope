"""Read-only sample audit; writes only a small receipt in this notes directory."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    scratch = Path("/scratch/09032/saini_2/constancy_rope")
    notes = Path(__file__).resolve().parent
    project = notes.parent.parent
    manifest = json.loads((notes / "dev2_manifest.json").read_text())
    for name, expected in manifest["code_sha256"].items():
        assert hashlib.sha256((project / name).read_bytes()).hexdigest() == expected, name
    dev = json.loads((notes / "dev2_generation_check.json").read_text())
    prompts = [json.loads(line) for line in (scratch / "prompts/aesthetic4k_eval.jsonl").read_text().splitlines()]
    expected = {row["id"]: row["prompt"] for row in prompts[7:195]}
    root = scratch / "out_krea/main2/sega_ctm/4096x4096"
    baseline = {}
    for path in (scratch / "out_krea/main/sega_official/4096x4096").glob("*.json"):
        meta = json.loads(path.read_text())
        baseline[meta["config"]["prompt_id"]] = meta
    checked = []
    seen = set()
    for path in sorted(root.glob("*.json")):
        if not path.with_suffix(".png").exists():
            continue
        meta = json.loads(path.read_text())
        config = meta["config"]
        pid = config["prompt_id"]
        assert pid in expected and pid not in seen, pid
        seen.add(pid)
        assert meta["complete"] and config["prompt"] == expected[pid], pid
        requirements = dict(method="sega_ctm", eta=1.0, temperature="none", seed=0,
                            steps=28, guidance=4.5, res="4096x4096",
                            implementation_sha256=dev["implementation_sha256"],
                            profile_sha256=dev["profile_sha256"])
        assert all(config.get(key) == value for key, value in requirements.items()), pid
        assert config["rope"]["method"] == "sega_official", pid
        ref = baseline[pid]
        for key in ("prompt", "seed", "derived_seed", "steps", "guidance", "res", "model",
                    "model_revision", "dtype", "batch_size", "diffusers", "torch", "grid"):
            assert config[key] == ref["config"][key], (pid, key)
        assert meta["sigmas"] == ref["sigmas"] and len(meta["sigmas"]) == 28, pid
        with path.with_suffix(".png").open("rb") as stream:
            header = stream.read(24)
        assert header[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", header[16:24]) == (4096, 4096), pid
        gains = meta["gains"]
        assert len(gains) == 28, pid
        maximum = 0.0
        for i, step in enumerate(gains):
            total = np.asarray(step["log_g"], dtype=float)
            official = np.asarray(step["official_log_g"], dtype=float)
            assert total.shape == official.shape == (128,), pid
            assert np.isfinite(total).all() and np.isfinite(official).all(), pid
            assert step["sigma"] == meta["sigmas"][i], pid
            assert np.array_equal(total[::2], total[1::2]), pid
            assert np.all(total[:16] == 0), pid
            correction = total - official
            assert np.max(np.abs(correction)) <= 1.000001, pid
            if i == 0:
                assert step["first_step_no_correction"] and np.allclose(correction, 0, atol=1e-7), pid
            maximum = max(maximum, float(np.max(np.abs(correction))))
        assert maximum > 1e-5, pid
        checked.append(dict(prompt_id=pid, seconds=meta["seconds"], max_absolute_logit_correction=maximum,
                            sidecar_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if args.complete:
        assert seen == set(expected), (len(seen), len(expected))
    receipt = dict(checked_at=datetime.now().astimezone().isoformat(), status="passed",
                   complete=seen == set(expected), checked_images=len(checked), expected_images=188,
                   heldout_images=sum(row["prompt_id"] != prompts[7]["id"] for row in checked),
                   checks="Frozen source/profile, matched prompt/seed/model/software/executed sigmas, PNG dimensions, finite paired gains, first-step parity, bounded active correction", samples=checked)
    (notes / "main2_generation_check.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "samples"}))


if __name__ == "__main__":
    main()
