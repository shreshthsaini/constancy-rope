#!/usr/bin/env python3
"""Export Aesthetic-4K eval captions and matched reference paths.

Official layout: eval/size_4096/metadata.jsonl, with file_name and text.
Source: https://huggingface.co/datasets/zhang0jhon/Aesthetic-4K
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote


DATASET = "zhang0jhon/Aesthetic-4K"
SCRATCH = Path("/scratch/09032/saini_2/constancy_rope")
DEFAULT_HF_HOME = "/scratch/09032/saini_2/tmp/hf"


def parse_metadata(path: Path) -> list[dict]:
    """Read imagefolder rows without decoding or copying reference images."""
    rows = []
    filenames = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            file_name, prompt = row.get("file_name"), row.get("text")
            if not isinstance(file_name, str) or not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: expected file_name and nonempty text")
            relative = PurePosixPath(file_name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in file_name:
                raise ValueError(f"{path}:{line_number}: unsafe file_name {file_name!r}")
            if file_name in filenames:
                raise ValueError(f"{path}:{line_number}: duplicate file_name {file_name!r}")
            filenames.add(file_name)
            rows.append({"file_name": file_name, "prompt": prompt.strip()})
    if not rows:
        raise ValueError(f"No captions in {path}")
    return rows


def prompt_id(file_name: str, size: int) -> str:
    stem = PurePosixPath(file_name).stem
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
        raise ValueError(f"Unsupported image filename stem {stem!r}")
    return f"a4k_eval_{size}_{stem}"


def snapshot_revision(path: Path, fallback: str) -> str:
    parts = path.absolute().parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and re.fullmatch(r"[a-f0-9]{40}", parts[index + 1]):
            return parts[index + 1]
    return fallback


def hub_file(filename: str, revision: str, offline: bool) -> Path:
    """Prefer the shared cache and download only a requested missing file."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    cache = Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)) / "hub"
    kwargs = dict(repo_id=DATASET, repo_type="dataset", filename=filename,
                  revision=revision, cache_dir=str(cache))
    try:
        return Path(hf_hub_download(**kwargs, local_files_only=True))
    except LocalEntryNotFoundError:
        if offline:
            raise FileNotFoundError(f"Dataset file is not cached: {filename} at {revision}") from None
        return Path(hf_hub_download(**kwargs, local_files_only=False))


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def export(args: argparse.Namespace) -> dict:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.start < 0:
        raise ValueError("--start must be nonnegative")
    if args.out.absolute() == args.references.absolute():
        raise ValueError("Prompt and reference manifests must be different files")
    prefix = f"eval/size_{args.size}"
    metadata = args.metadata or hub_file(f"{prefix}/metadata.jsonl", args.revision, args.offline)
    metadata = metadata.absolute()
    revision = snapshot_revision(metadata, args.revision)
    rows = parse_metadata(metadata)
    stop = args.start + args.limit if args.limit is not None else None
    rows = rows[args.start:stop]
    if not rows:
        raise ValueError("Selected prompt range is empty")
    prompts, references, ids = [], [], set()
    for row in rows:
        key = prompt_id(row["file_name"], args.size)
        if key in ids:
            raise ValueError(f"Image stems produce duplicate prompt ID: {key}")
        ids.add(key)
        relative = f"{prefix}/{row['file_name']}"
        image = metadata.parent / row["file_name"]
        if not image.is_file() or image.stat().st_size == 0:
            image = hub_file(relative, revision, args.offline)
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty reference image: {image}")
        image = image.absolute()
        provenance = {
            "dataset": DATASET,
            "dataset_revision": revision,
            "split": "eval",
            "reference_size": args.size,
            "source_file_name": relative,
            "source_url": f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{quote(relative)}",
        }
        prompts.append({"id": key, "prompt": row["prompt"],
                        "reference_image": str(image), **provenance})
        references.append({"id": key, "prompt": row["prompt"], "path": str(image),
                           "image": str(image), **provenance})
    atomic_jsonl(args.out, prompts)
    atomic_jsonl(args.references, references)
    return {"count": len(prompts), "dataset": DATASET, "revision": revision,
            "metadata": str(metadata), "prompts": str(args.out.absolute()),
            "references": str(args.references.absolute())}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", type=Path, default=SCRATCH / "prompts/aesthetic4k_eval.jsonl")
    result.add_argument("--references", type=Path, default=SCRATCH / "prompts/aesthetic4k_refs.jsonl")
    result.add_argument("--size", type=int, choices=(2048, 4096), default=4096)
    result.add_argument("--revision", default="main", help="HF dataset revision; the resolved commit is recorded")
    result.add_argument("--metadata", type=Path, help="Use a local imagefolder metadata.jsonl")
    result.add_argument("--offline", action="store_true", help="Require all selected files to be cached locally")
    result.add_argument("--start", type=int, default=0, help="First metadata row, preserving dataset order")
    result.add_argument("--limit", type=int, help="Maximum number of selected eval captions")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = export(args)
    except (ImportError, OSError, ValueError) as error:
        parser().error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
