#!/usr/bin/env python3
"""Generate one resumable FLUX campaign chunk."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crope.utils import read_prompts, resolution, runtime_environment


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("base", "pi", "ntk", "ntk2", "yarn", "dype", "yarn_temp",
                   "sega", "sega_official", "sega_static", "sega_official_static", "constancy", "temp_only", "constancy_nogate", "constancy_sref"))
    p.add_argument("--res", type=resolution, required=True, metavar="HxW")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--sega-static-profile", type=Path)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--target", choices=("profile", "slope"), default="profile")
    p.add_argument("--temperature", choices=("logN", "sref", "none"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-offload", action="store_true", help="Use only if the ordinary GPU run exhausts VRAM")
    p.add_argument("--allow-download", action="store_true", help="Allow missing FLUX cache files to download")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts) or args.steps <= 0:
        raise ValueError("Require 0 <= start < end <= number of prompts and positive steps")
    from crope.runtime import FluxExperiment, load_pipeline
    pipe = load_pipeline(args.device, args.cpu_offload, args.allow_download)
    experiment = FluxExperiment(pipe, args.method, args.res, args.profile, args.sega_static_profile,
                                args.eta, args.target, args.temperature)
    try:
        for row in prompts[args.start:end]:
            experiment.sample(row, args.out, args.seed, args.steps, args.guidance)
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
