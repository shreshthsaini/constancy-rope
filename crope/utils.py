"""Small, dependency-light experiment and provenance helpers."""

import hashlib
import json
import os
from pathlib import Path
import re

SCRATCH = Path("/scratch/09032/saini_2/constancy_rope")
HF_HOME = "/scratch/09032/saini_2/tmp/hf"


def runtime_environment():
    # Set before importing numpy or torch. Callers can request fewer threads.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(min(8, max(1, int(os.environ.get(name, "8")))))
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("TORCH_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("MPLCONFIGDIR", str(SCRATCH / "cache" / "matplotlib"))
    os.environ.setdefault("WANDB_DIR", str(SCRATCH / "wandb"))
    os.environ.setdefault("WANDB_PROJECT", "constancy-rope")


def resolution(value):
    if isinstance(value, (list, tuple)):
        h, w = map(int, value)
    else:
        match = re.fullmatch(r"(\d+)[xX](\d+)", value)
        if match is None:
            raise ValueError("Resolution must be HxW, for example 4096x4096")
        h, w = map(int, match.groups())
    if min(h, w) < 64 or h % 16 or w % 16:
        raise ValueError("Both dimensions must be multiples of 16 and at least 64")
    return h, w


def read_prompts(path):
    rows = []
    seen = set()
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["id"])
            if key in seen or not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError(f"Duplicate ID or invalid prompt: {key}")
            seen.add(key)
            rows.append({**row, "id": key})
    return rows


def image_seed(prompt_id, seed):
    payload = json.dumps([str(prompt_id), int(seed)], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def image_stem(prompt_id, seed):
    key = str(prompt_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80].strip(".") or "prompt"
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    return f"{safe}_{digest}_seed{int(seed)}"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def file_sha256(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

