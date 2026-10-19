#!/usr/bin/env python3
"""Prepare the frozen fresh150 TRAIN sample without generating model images.

Run ``candidates`` first, review its caption-only JSONL, write an audit JSON with
one decision per reviewed source_file_name, then run ``finalize --audit PATH``.
The sampler traverses one fixed permutation and never edits source caption text.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import unicodedata

import numpy as np

DATASET = "zhang0jhon/Aesthetic-4K"
REVISION = "8c5d5cb8b94230ff897d87bf060451257d1c7bf8"
WORK = Path(__file__).resolve().parents[1]
SCRATCH = Path("/scratch/09032/saini_2/constancy_rope")
HF_HOME = Path("/scratch/09032/saini_2/tmp/hf")
NOTES = WORK / ".agent/notes/strengthen"
CAPTION_KEYS = {"prompt", "caption", "text", "positive_prompt", "negative_prompt"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "third_party", "wandb",
             "telemetry", "test_runs", "test_runs_strengthen"}
HUMAN = re.compile(
    r"\b(?:people|person|persons|human|humans|man|men|woman|women|girl|girls|boy|boys|"
    r"child|children|kid|kids|baby|babies|infant|infants|teen\w*|toddler\w*|"
    r"portrait\w*|face|faces|facial|selfie\w*|crowd\w*|spectator\w*|pedestrian\w*|"
    r"tourist\w*|visitor\w*|worker\w*|farmer\w*|soldier\w*|rider\w*|cyclist\w*|"
    r"runner\w*|hiker\w*|climber\w*|skier\w*|surfer\w*|fisherman|fishermen|"
    r"fisherwoman|fisherwomen|player\w*|athlete\w*|musician\w*|dancer\w*|"
    r"singer\w*|performer\w*|couple\w*|family|families|bride\w*|groom\w*|"
    r"wearing|wears|dressed|clothed|smiling|smiles|smile|hands?|fingers?|"
    r"hair|hairstyle\w*|beard\w*|moustache\w*|mustache\w*|makeup|lipstick|"
    r"lips?|eyelash\w*|eyebrow\w*|skin|tattoo\w*|nail\s+polish|"
    r"angel\w*|fairy|fairies|mermaid\w*|goddess\w*|god|gods|buddha|jesus|"
    r"king|queen|prince|princess|monk\w*|nun|nuns|worshipper\w*|"
    r"commuter\w*|passenger\w*|driver\w*|pilot\w*|audience|student\w*|"
    r"sitting|standing|walking|kneeling|posing|poses)\b", re.I)
UNSAFE = re.compile(
    r"\b(?:nude\w*|naked|sexual\w*|sex|porn\w*|erotic\w*|lingerie|bikini\w*|"
    r"topless|breasts?|nipples?|genital\w*|penis|vagina|buttocks|fetish\w*|"
    r"blood\w*|gore|gory|gruesome|severed|decapitat\w*|disembowel\w*|"
    r"corpse\w*|cadaver\w*|murder\w*|suicide\w*|tortur\w*|rape\w*|"
    r"gun\w*|rifle\w*|pistol\w*|weapon\w*|bomb\w*|grenade\w*|"
    r"cocaine|heroin|methamphetamine|cannabis|marijuana|swastika\w*|nazi\w*)\b", re.I)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value, *, jsonl=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        data = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value)
    else:
        data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() == data:
            return
        raise FileExistsError(f"Refusing to replace frozen artifact: {path}")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)


def strings(value, key=""):
    if isinstance(value, str) and key.casefold() in CAPTION_KEYS:
        yield value
    elif isinstance(value, dict):
        for name, child in value.items():
            yield from strings(child, name)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child, "prompt" if key in {"prompts", "captions"} else key)


def collect_exclusions(roots: list[Path], skip_paths=()) -> tuple[set[str], dict]:
    """Audit JSON/JSONL/CSV prompt fields, including all generation sidecars."""
    excluded, sources, errors, scanned = set(), [], [], 0
    skip_paths = {p.absolute() for p in skip_paths}
    seen_paths = set()
    for root in roots:
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                path = Path(current) / name
                if path.absolute() in skip_paths or path in seen_paths:
                    continue
                seen_paths.add(path)
                if path.suffix not in {".json", ".jsonl", ".csv"}:
                    continue
                if "strengthen" in path.parts or name.startswith("fresh150"):
                    continue
                scanned += 1
                values = []
                try:
                    with path.open(encoding="utf-8") as handle:
                        if path.suffix == ".jsonl":
                            for line in handle:
                                if line.strip():
                                    values.extend(strings(json.loads(line)))
                        elif path.suffix == ".csv":
                            for row in csv.DictReader(handle):
                                values.extend(strings(row))
                        else:
                            values.extend(strings(json.load(handle)))
                except (UnicodeError, json.JSONDecodeError, OSError) as error:
                    errors.append({"path": str(path), "error": str(error)})
                    continue
                values = [norm(value) for value in values if value.strip()]
                if values:
                    excluded.update(values)
                    sources.append({"path": str(path), "sha256": sha(path),
                                    "caption_entries": len(values), "unique_captions": len(set(values))})
    normalized_hash = hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()
    return excluded, {"created_utc": now(), "roots": [str(p) for p in roots],
                      "scanned_files": scanned, "source_files": sources,
                      "unique_caption_count": len(excluded), "normalized_set_sha256": normalized_hash,
                      "parse_errors": errors, "skipped_directory_names": sorted(SKIP_DIRS),
                      "fields": sorted(CAPTION_KEYS),
                      "normalization": "Unicode NFKC, casefold, collapsed whitespace"}


def load_rows(metadata: Path) -> list[dict]:
    rows, identifiers = [], set()
    with metadata.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            relative, caption = item["file_name"], item["text"]
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
                raise ValueError(f"Unsafe source path: {relative}")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"Empty caption at row {index}")
            identifier = "train/" + relative
            if identifier in identifiers:
                raise ValueError(f"Duplicate source filename: {identifier}")
            identifiers.add(identifier)
            rows.append({"source_file_name": identifier, "source_row": index, "prompt": caption})
    return sorted(rows, key=lambda row: (row["source_file_name"], row["source_row"]))


def filter_reason(caption: str) -> str | None:
    if match := UNSAFE.search(caption):
        return "unsafe_keyword:" + match.group(0).casefold()
    if match := HUMAN.search(caption):
        return "conservative_person_scene:" + match.group(0).casefold()
    return None


def select(rows: list[dict], excluded: set[str], count: int, audit: dict | None = None):
    selected, rejected, seen = [], [], set()
    for rank, index in enumerate(np.random.default_rng(2026).permutation(len(rows))):
        row = {**rows[int(index)], "permutation_rank": rank, "sorted_source_index": int(index)}
        normalized = norm(row["prompt"])
        reason = "prior_project_use" if normalized in excluded else None
        reason = reason or ("duplicate_caption" if normalized in seen else None)
        reason = reason or filter_reason(row["prompt"])
        if reason is None and audit is not None:
            decision = audit.get(row["source_file_name"])
            if decision is None:
                raise ValueError(f"Caption not manually reviewed: {row['source_file_name']}")
            if decision["decision"] == "reject":
                reason = "manual:" + decision["reason"]
            elif decision["decision"] != "accept":
                raise ValueError(f"Unknown audit decision: {decision}")
        if reason:
            rejected.append({**row, "reason": reason})
        else:
            selected.append(row)
            seen.add(normalized)
            if len(selected) == count:
                return selected, rejected
    raise ValueError(f"Only {len(selected)} eligible unique captions")


def hub_file(filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    return Path(hf_hub_download(DATASET, filename=filename, repo_type="dataset", revision=REVISION,
                               cache_dir=str(HF_HOME / "hub")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("candidates", "finalize"))
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--candidate-count", type=int, default=180)
    args = parser.parse_args()
    metadata = hub_file("train/metadata.jsonl")
    rows = load_rows(metadata)
    if args.phase == "candidates":
        excluded, receipt = collect_exclusions([WORK, SCRATCH])
        # The normalized values are retained to reproduce exclusion decisions.
        receipt["normalized_captions"] = sorted(excluded)
        write_once(NOTES / "fresh_selection_exclusions.json", receipt)
        chosen, rejected = select(rows, excluded, args.candidate_count)
        write_once(NOTES / "fresh_selection_candidates.jsonl", chosen, jsonl=True)
        write_once(NOTES / "fresh_selection_candidate_rejections.jsonl", rejected, jsonl=True)
        print(json.dumps({"candidates": len(chosen), "rejections": len(rejected),
                          "dataset_rows": len(rows), "prior_captions": len(excluded),
                          "parse_errors": receipt["parse_errors"]}, indent=2))
        return
    if args.audit is None:
        parser.error("finalize requires --audit")
    exclusion_path = NOTES / "fresh_selection_exclusions.json"
    exclusions = json.loads(exclusion_path.read_text())
    audit_document = json.loads(args.audit.read_text())
    decisions = {r["source_file_name"]: r for r in audit_document["decisions"]}
    chosen, rejected = select(rows, set(exclusions["normalized_captions"]), 150, decisions)
    from PIL import Image
    prompts, references, downloaded = [], [], []
    for index, row in enumerate(chosen):
        image = hub_file(row["source_file_name"]).absolute()
        with Image.open(image) as handle:
            dimensions = list(handle.size)
            handle.verify()
        image_hash = sha(image)
        stem = PurePosixPath(row["source_file_name"]).stem
        folder = PurePosixPath(row["source_file_name"]).parent.name
        key = f"a4k_fresh150_train_{folder}_{stem}"
        provenance = {**row, "id": key, "dataset": DATASET, "dataset_revision": REVISION,
                      "split": "train", "reference_dimensions": dimensions,
                      "reference_sha256": image_hash}
        prompts.append({**provenance, "reference_image": str(image)})
        references.append({**provenance, "path": str(image), "image": str(image)})
        downloaded.append({"path": str(image), "bytes": image.stat().st_size, "sha256": image_hash})
        print(f"reference {index + 1}/150 {key}", flush=True)
    prompt_path, refs_path = WORK / "prompts/fresh150.jsonl", WORK / "prompts/fresh150_refs.jsonl"
    write_once(prompt_path, prompts, jsonl=True)
    write_once(refs_path, references, jsonl=True)
    write_once(SCRATCH / "prompts/fresh150.jsonl", prompts, jsonl=True)
    write_once(SCRATCH / "prompts/fresh150_refs.jsonl", references, jsonl=True)
    write_once(NOTES / "fresh_selection_rejections.jsonl", rejected, jsonl=True)
    receipt = {"completed_utc": now(), "count": len(prompts), "dataset": DATASET,
               "dataset_revision": REVISION, "metadata_path": str(metadata),
               "metadata_sha256": sha(metadata), "metadata_rows": len(rows),
               "selection": "sorted source_file_name/source_row; default_rng(2026).permutation; first 150 eligible",
               "numpy_version": np.__version__, "prompt_path": str(prompt_path),
               "prompt_sha256": sha(prompt_path), "reference_manifest_path": str(refs_path),
               "reference_manifest_sha256": sha(refs_path), "reference_files": downloaded,
               "reference_bytes": sum(row["bytes"] for row in downloaded),
               "exclusion_receipt": str(exclusion_path), "exclusion_receipt_sha256": sha(exclusion_path),
               "audit_path": str(args.audit.absolute()), "audit_sha256": sha(args.audit),
               "script_sha256": sha(Path(__file__)), "rejected_candidates": len(rejected),
               "final_permutation_rank": chosen[-1]["permutation_rank"],
               "image_generation_performed": False,
               "limitations": ["Caption-only filter conservatively excludes person scenes; this changes the sampled population.",
                               "Fresh means unused in project records; model pretraining overlap is not assessed."]}
    write_once(NOTES / "fresh_selection_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "reference_files"}, indent=2))


if __name__ == "__main__":
    main()
