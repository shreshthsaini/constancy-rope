#!/usr/bin/env python3
"""Build native Constancy profiles or target-resolution SEGA-static profiles."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crope.utils import image_stem, read_prompts, resolution, runtime_environment


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("native", "sega-static"), required=True)
    p.add_argument("--prompts", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--out", type=Path, default=Path("/scratch/09032/saini_2/constancy_rope/calib/flux_dev_profile.npz"))
    p.add_argument("--runs", type=Path, nargs="+", help="Completed target-resolution SEGA run directories")
    p.add_argument("--res", type=resolution, metavar="HxW", help="Required for sega-static; native is always1024x1024")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--allow-download", action="store_true")
    return p


def static_profile(runs, res, out):
    import numpy as np
    modulators, sources, identities = [], [], set()
    settings = set()
    for path in sorted({p for directory in runs for p in directory.rglob("*.json")}):
        record = json.loads(path.read_text())
        config = record.get("config", {})
        if config.get("method") != "sega" or config.get("res") != f"{res[0]}x{res[1]}":
            continue
        image = Path(record.get("image", path.with_suffix(".png")))
        if not record.get("complete") or not image.is_file():
            continue
        identity = (str(config["prompt_id"]), config["seed"])
        if identity in identities:
            raise ValueError(f"Duplicate prompt and seed in static calibration: {identity}")
        identities.add(identity)
        gains = record["gains"]
        if len(gains) != config["steps"]:
            raise ValueError(f"Incomplete gain trajectory: {path}")
        values = np.asarray([g["modulator"] for g in gains], dtype=np.float64)
        if values.shape[1:] != (2, 28) or not np.isfinite(values).all():
            raise ValueError(f"Invalid SEGA M vectors: {path}")
        modulators.append(values)
        settings.add((config.get("model"), config.get("model_revision"), config["steps"], config["guidance"]))
        sources.append(str(path.resolve()))
    if not modulators:
        raise ValueError("No completed SEGA images at the requested resolution")
    if len(settings) > 1:
        raise ValueError("SEGA-static calibration mixes models or sampling settings")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, mean_modulator=np.concatenate(modulators).mean(0), grid=np.asarray(res) // 16,
                 axis_order=np.asarray(["height", "width"]), source_json=np.asarray(sources),
                 prompt_ids=np.asarray([p for p, _ in sorted(identities)]), n_images=len(modulators),
                 n_steps=sum(len(m) for m in modulators), config_json=json.dumps(list(settings)[0]))
    os.replace(temporary, out)
    print(f"saved {out}: {len(modulators)} images, {sum(map(len, modulators))} steps", flush=True)


def native_profile(args):
    import numpy as np
    import torch
    from crope.runtime import FluxExperiment, fingerprint, load_pipeline
    from crope.spectral import NativeProfile, axis_power

    prompts = read_prompts(args.prompts)
    if args.start < 0 or args.count <= 0 or args.start + args.count > len(prompts):
        raise ValueError("Calibration range must be nonempty and contained in the prompt file")
    selected = prompts[args.start:args.start + args.count]
    pipe = load_pipeline(args.device, args.cpu_offload, args.allow_download)
    experiment = FluxExperiment(pipe, "base", (1024, 1024))
    trace_dir = args.out.parent / (args.out.stem + "_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    logs = [[], []]
    frequencies = None
    sources = []
    try:
        for row in selected:
            path = trace_dir / f"{image_stem(row['id'], args.seed)}.npz"
            config = experiment.config(row, args.seed, args.steps, args.guidance)
            digest = fingerprint(config)
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if str(data["config_fingerprint"]) != digest:
                        raise ValueError(f"Existing calibration trace has different settings: {path}")
                    trace = {key: data[key].copy() for key in data.files}
                print(f"skip native trace {row['id']}", flush=True)
            else:
                sigmas, powers = [], [[], []]
                def observe(step, sigma, xt, x0):
                    nonlocal frequencies
                    axes = axis_power(x0)
                    frequencies = tuple(a.frequency.cpu().numpy() for a in axes)
                    sigmas.append(sigma)
                    for a, power in enumerate(axes):
                        powers[a].append(power.power.clamp_min(1e-30).log().cpu().numpy())
                _, seconds, _ = experiment.generate(row, args.seed, args.steps, args.guidance,
                                                     output_type="latent", observer=observe)
                trace = {"sigma": np.asarray(sigmas), "config_fingerprint": digest,
                         "config_json": json.dumps(config), "seconds": seconds,
                         **{f"frequency_{a}": frequencies[i] for i, a in enumerate(("h", "w"))},
                         **{f"log_power_{a}": np.stack(powers[i]) for i, a in enumerate(("h", "w"))}}
                temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
                with temporary.open("wb") as handle:
                    np.savez(handle, **trace)
                os.replace(temporary, path)
                print(f"native {row['id']} seconds={seconds:.3f}", flush=True)
            late = trace["sigma"] < 0.3
            if not late.any():
                raise ValueError("No native steps with sigma<0.3; increase --steps")
            frequencies = tuple(trace[f"frequency_{a}"] for a in ("h", "w"))
            for i, a in enumerate(("h", "w")):
                logs[i].append(trace[f"log_power_{a}"][late])
            sources.append(str(path.resolve()))
    finally:
        experiment.close()
    profile = NativeProfile(tuple(torch.from_numpy(f) for f in frequencies),
                            tuple(torch.from_numpy(np.concatenate(values).mean(0)) for values in logs))
    profile.save(args.out, prompt_ids=np.asarray([r["id"] for r in selected]),
                 source_traces=np.asarray(sources), n_prompts=len(selected),
                 n_late_steps=sum(len(v) for v in logs[0]), native_grid=np.asarray([64, 64]),
                 channels=64, late_sigma_threshold=0.3,
                 config_json=json.dumps({"model": "black-forest-labs/FLUX.1-dev", "steps": args.steps,
                                         "guidance": args.guidance, "seed": args.seed,
                                         "power": "channel-summed unnormalized FFT2 axis-marginal"}))
    print(f"saved native profile {args.out}: {len(selected)} prompts", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    if args.mode == "native":
        if args.prompts is None or (args.res is not None and args.res != (1024, 1024)):
            raise ValueError("Native mode requires --prompts and resolution1024x1024")
        native_profile(args)
    else:
        if not args.runs or args.res is None:
            raise ValueError("SEGA-static mode requires --runs and --res")
        static_profile(args.runs, args.res, args.out)


if __name__ == "__main__":
    main()
