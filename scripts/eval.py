#!/usr/bin/env python3
"""Evaluate completed sample PNG/JSON pairs and bootstrap by prompt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Root containing sample PNG files and JSON sidecars")
    p.add_argument("--prompts", required=True, help="Prompt JSONL with id and prompt")
    p.add_argument("--references", help="Reference JSONL with id and path")
    p.add_argument("--reference-method", default="sega_official")
    p.add_argument("--res", help="Optional resolution filter, for example 4096x4096")
    p.add_argument("--out", required=True, help="Evaluation directory on scratch, also holds resumable caches")
    p.add_argument("--metrics", default="all", help="all (HPSv3 optional) or comma list: clip,imagereward,pickscore,hpsv3,musiq,clipiqa,fid,kid,patch_fid,patch_kid,spectrum,tiling")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--crop-seed", type=int, default=0)
    p.add_argument("--crops", type=int, default=4, help="Fixed random 1024 crops per image, shared across methods")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--allow-missing", action="store_true", help="Exit successfully with explicitly reported unavailable metrics")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging explicitly, useful for CPU numerical tests")
    return p


def main():
    args = parser().parse_args()
    if args.bootstrap < 0 or args.crops < 1 or not 1 <= args.threads <= 8:
        raise SystemExit("--bootstrap must be nonnegative, --crops positive, and --threads between 1 and 8")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ.setdefault("HF_HOME", "/scratch/09032/saini_2/tmp/hf")
    os.environ.setdefault("TORCH_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("WANDB_DIR", "/scratch/09032/saini_2/constancy_rope/wandb")
    os.environ.setdefault("WANDB_PROJECT", "constancy-rope")
    from crope.metrics.evaluate import Evaluation

    return Evaluation(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
