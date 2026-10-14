"""Read sample sidecars and keep evaluation checkpoints atomic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def file_signature(path: str | Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "tolist"):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def normalize_resolution(value, config: dict) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{int(value[0])}x{int(value[1])}"
    if value is not None:
        return str(value).lower().replace(" ", "")
    if "height" in config and "width" in config:
        return f"{int(config['height'])}x{int(config['width'])}"
    raise ValueError("Sample metadata lacks resolution")


def discover_samples(root: str | Path, prompts: list[dict], resolution: str | None = None) -> list[dict]:
    prompt_map = {str(row["id"]): row["prompt"] for row in prompts}
    result, seen = [], set()
    for sidecar in sorted(Path(root).rglob("*.json")):
        image = sidecar.with_suffix(".png")
        if not image.is_file():
            continue
        with sidecar.open() as handle:
            metadata = json.load(handle)
        if metadata.get("complete", True) is not True:
            continue
        config = metadata.get("config", metadata)
        if "method" not in config and "method" not in metadata:
            continue
        prompt_id = str(config.get("prompt_id", metadata.get("prompt_id", metadata.get("id", ""))))
        if prompt_id not in prompt_map:
            continue
        res = normalize_resolution(config.get("res", config.get("resolution", metadata.get("res"))), config)
        if resolution and res != resolution.lower():
            continue
        dimensions = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", res)
        if dimensions is None:
            raise ValueError(f"Invalid HxW resolution {res!r} in {sidecar}")
        from PIL import Image

        with Image.open(image) as opened:
            if opened.size != (int(dimensions[2]), int(dimensions[1])):
                raise ValueError(f"PNG size {opened.size} does not match sidecar resolution {res}: {image}")
        method = str(config.get("method", metadata.get("method")))
        seed = config.get("seed", metadata.get("seed"))
        if seed is None:
            raise ValueError(f"Missing seed in {sidecar}")
        identity = (method, res, prompt_id, str(seed))
        if identity in seen:
            raise ValueError(f"Duplicate sample identity {identity}: {sidecar}")
        seen.add(identity)
        prompt = str(config.get("prompt", metadata.get("prompt", prompt_map[prompt_id])))
        if prompt != prompt_map[prompt_id]:
            raise ValueError(f"Prompt text differs from manifest for {prompt_id}: {sidecar}")
        result.append({"method": method, "res": res, "prompt_id": prompt_id, "seed": seed,
                       "prompt": prompt, "path": str(image.resolve()),
                       "config_hash": digest(config),
                       "seconds": metadata.get("seconds", metadata.get("timing", {}).get("seconds")),
                       "sidecar": str(sidecar.resolve())})
    if not result:
        raise ValueError(f"No completed PNG/JSON samples matching prompts under {root}")
    return result


def load_references(manifest: str | Path) -> list[dict]:
    manifest = Path(manifest)
    references = []
    for row in read_jsonl(manifest):
        name = row.get("path", row.get("image_path", row.get("image", row.get("reference_image"))))
        if isinstance(name, dict):
            name = name.get("path")
        if not name:
            raise ValueError(f"Reference {row['id']} has no local image path")
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        references.append({"prompt_id": str(row["id"]), "path": str(path.resolve())})
    return references
