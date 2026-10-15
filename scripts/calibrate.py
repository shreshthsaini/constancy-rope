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
    p.add_argument("--mode", choices=("native", "native-rebuild", "sega-static"), required=True)
    p.add_argument("--traces", type=Path, help="Existing native trace directory for CPU-only native-rebuild")
    p.add_argument("--prompts", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--out", type=Path, default=Path("/scratch/09032/saini_2/constancy_rope/calib/flux_dev_profile.npz"))
    p.add_argument("--runs", type=Path, nargs="+", help="Completed target-resolution SEGA run directories")
    p.add_argument("--method", choices=("sega", "sega_official"), default="sega",
                   help="Source method for sega-static calibration")
    p.add_argument("--res", type=resolution, metavar="HxW", help="Required for sega-static; native is always1024x1024")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--allow-download", action="store_true")
    return p


def static_profile(runs, res, out, method="sega"):
    import numpy as np
    if method not in ("sega", "sega_official"):
        raise ValueError("Static calibration requires sega or sega_official runs")
    modulators, sources, identities = [], [], set()
    settings = set()
    for path in sorted({p for directory in runs for p in directory.rglob("*.json")}):
        record = json.loads(path.read_text())
        config = record.get("config", {})
        if config.get("method") != method or config.get("res") != f"{res[0]}x{res[1]}":
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
        if method == "sega_official":
            values = np.asarray([g["controller_log_g"] for g in gains], dtype=np.float64)
            with np.errstate(over="ignore"):
                values = np.exp(values / 2)
            if values.shape[1:] != (128,) or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Invalid released-SEGA q/k gains: {path}")
            if not np.array_equal(values[:, 0::2], values[:, 1::2]):
                raise ValueError(f"Released-SEGA q/k gains must match within rotary pairs: {path}")
        else:
            values = np.asarray([g["modulator"] for g in gains], dtype=np.float64)
            if values.shape[1:] != (2, 28) or not np.isfinite(values).all():
                raise ValueError(f"Invalid SEGA M vectors: {path}")
        modulators.append(values)
        settings.add((config.get("model"), config.get("model_revision"), config["steps"], config["guidance"]))
        sources.append(str(path.resolve()))
    if not modulators:
        raise ValueError(f"No completed SEGA images for {method} at the requested resolution")
    if len(settings) > 1:
        raise ValueError("SEGA-static calibration mixes models or sampling settings")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    key = "mean_qk_scale" if method == "sega_official" else "mean_modulator"
    with temporary.open("wb") as handle:
        np.savez(handle, **{key: np.concatenate(modulators).mean(0)}, grid=np.asarray(res) // 16,
                 source_method=method,
                 axis_order=np.asarray(["height", "width"]), source_json=np.asarray(sources),
                 prompt_ids=np.asarray([p for p, _ in sorted(identities)]), n_images=len(modulators),
                 n_steps=sum(len(m) for m in modulators), config_json=json.dumps(list(settings)[0]))
    os.replace(temporary, out)
    print(f"saved {out}: {len(modulators)} images, {sum(map(len, modulators))} steps", flush=True)


def aggregate_native_traces(paths, out):
    """Keep the late-step reduction unchanged and also average each native step."""
    import numpy as np
    import torch
    from crope.spectral import POWER_NORMALIZATION, NativeProfile

    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("No native .npz traces found")
    late_threshold = float(os.environ.get("CROPE_LATE_SIGMA", "0.3"))
    if not 0 < late_threshold <= 1:
        raise ValueError("CROPE_LATE_SIGMA must be in (0,1]")
    logs, trajectories = [[], []], [[], []]
    frequencies, sigmas, settings = None, None, None
    prompt_ids, identities = [], set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            trace = {key: data[key].copy() for key in data.files}
        required = {"config_json", "sigma", "frequency_h", "frequency_w", "log_power_h", "log_power_w"}
        if missing := required - trace.keys():
            raise ValueError(f"Incomplete native trace {path}: missing {sorted(missing)}")
        config = json.loads(str(trace["config_json"]))
        if config.get("power_normalization") != POWER_NORMALIZATION:
            raise ValueError(f"Native trace uses incompatible power normalization; recalibrate it: {path}")
        current_settings = {key: config.get(key) for key in
                            ("model", "model_revision", "steps", "guidance", "seed", "grid", "scheduler")}
        if settings is not None and current_settings != settings:
            raise ValueError(f"Native traces mix models or sampling settings: {path}")
        settings = current_settings
        identity = (str(config["prompt_id"]), config["seed"])
        if identity in identities:
            raise ValueError(f"Duplicate prompt and seed in native traces: {identity}")
        identities.add(identity)
        prompt_ids.append(identity[0])
        sigma = trace["sigma"]
        if sigma.ndim != 1 or len(sigma) != config["steps"]:
            raise ValueError(f"Incomplete native sigma trajectory: {path}")
        if sigmas is not None and not np.array_equal(sigma, sigmas):
            raise ValueError(f"Native trace sigma steps do not match: {path}")
        sigmas = sigma
        current_frequencies = tuple(trace[f"frequency_{a}"] for a in ("h", "w"))
        if frequencies is not None and any(not np.array_equal(f, g) for f, g in zip(frequencies, current_frequencies)):
            raise ValueError(f"Native trace frequencies do not match: {path}")
        frequencies = current_frequencies
        # Validate every trajectory before averaging, including finite values and ordering.
        powers = tuple(torch.from_numpy(trace[f"log_power_{a}"]) for a in ("h", "w"))
        NativeProfile(tuple(torch.from_numpy(f) for f in frequencies),
                      tuple(p.mean(0) for p in powers), torch.from_numpy(sigmas), powers)
        late = sigmas < late_threshold
        if not late.any():
            raise ValueError(f"No native steps with sigma<{late_threshold}; increase --steps")
        for i, a in enumerate(("h", "w")):
            logs[i].append(trace[f"log_power_{a}"][late])
            trajectories[i].append(trace[f"log_power_{a}"])
    profile = NativeProfile(tuple(torch.from_numpy(f) for f in frequencies),
                            tuple(torch.from_numpy(np.concatenate(values).mean(0)) for values in logs),
                            torch.from_numpy(sigmas),
                            tuple(torch.from_numpy(np.stack(values).mean(0)) for values in trajectories))
    profile.save(out, prompt_ids=np.asarray(prompt_ids), source_traces=np.asarray([str(p.resolve()) for p in paths]),
                 n_prompts=len(paths), n_late_steps=sum(len(v) for v in logs[0]), native_grid=np.asarray([64, 64]),
                 channels=64, late_sigma_threshold=late_threshold,
                 config_json=json.dumps({"model": config["model"], "steps": config["steps"],
                                         "guidance": config["guidance"], "seed": config["seed"],
                                         "power": POWER_NORMALIZATION}))
    print(f"saved native profile {out}: {len(paths)} prompts, {len(sigmas)} sigma steps", flush=True)


def rebuild_native_profile(traces, out):
    """Read traces only; preserve the original reduction order when available."""
    import numpy as np

    traces, out = Path(traces), Path(out)
    paths = sorted(traces.glob("*.npz"))
    if traces.resolve() in out.resolve().parents:
        raise ValueError("Rebuilt profile must be outside the read-only trace directory")
    original = traces.with_name(traces.name.removesuffix("_traces") + ".npz")
    if original.is_file():
        with np.load(original, allow_pickle=False) as data:
            ordered = [traces / Path(str(p)).name for p in data.get("source_traces", [])]
        if len(ordered) == len(paths) and set(ordered) == set(paths):
            paths = ordered
    aggregate_native_traces(paths, out)


def native_profile(args):
    import numpy as np
    from crope.runtime import FluxExperiment, fingerprint, load_pipeline
    from crope.spectral import POWER_NORMALIZATION, axis_power

    prompts = read_prompts(args.prompts)
    if args.start < 0 or args.count <= 0 or args.start + args.count > len(prompts):
        raise ValueError("Calibration range must be nonempty and contained in the prompt file")
    selected = prompts[args.start:args.start + args.count]
    pipe = load_pipeline(args.device, args.cpu_offload, args.allow_download)
    experiment = FluxExperiment(pipe, "base", (1024, 1024))
    trace_dir = args.out.parent / (args.out.stem + "_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    frequencies = None
    sources = []
    try:
        for row in selected:
            path = trace_dir / f"{image_stem(row['id'], args.seed)}.npz"
            config = experiment.config(row, args.seed, args.steps, args.guidance)
            config["power_normalization"] = POWER_NORMALIZATION
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
            sources.append(str(path.resolve()))
    finally:
        experiment.close()
    aggregate_native_traces(sources, args.out)


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    if args.mode == "native":
        if args.prompts is None or (args.res is not None and args.res != (1024, 1024)):
            raise ValueError("Native mode requires --prompts and resolution1024x1024")
        native_profile(args)
    elif args.mode == "native-rebuild":
        if args.traces is None:
            raise ValueError("native-rebuild requires --traces pointing to existing native traces")
        rebuild_native_profile(args.traces, args.out)
    else:
        if not args.runs or args.res is None:
            raise ValueError("SEGA-static mode requires --runs and --res")
        static_profile(args.runs, args.res, args.out, args.method)


if __name__ == "__main__":
    main()
