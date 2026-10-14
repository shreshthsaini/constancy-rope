#!/usr/bin/env python3
"""Report cached FLUX scheduler trajectories on CPU, without loading weights."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crope.utils import atomic_json, resolution, runtime_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", nargs="+", type=resolution,
                        default=[(1024, 1024), (2048, 4096), (3072, 3072), (4096, 4096)])
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    runtime_environment()
    from diffusers import FlowMatchEulerDiscreteScheduler
    from crope.schedule import diagnose_schedule
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="scheduler", local_files_only=True)
    reports = [diagnose_schedule(dict(scheduler.config), res, args.steps) for res in args.res]
    if args.out:
        atomic_json(args.out, reports)
    for r in reports:
        print(json.dumps({k: r[k] for k in ("res", "mu", "last_evaluated_sigma", "distinct_model_timesteps",
                                          "evaluations_below_sigma_0_3", "largest_euler_step")}))


if __name__ == "__main__":
    main()
