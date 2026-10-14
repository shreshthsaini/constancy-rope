#!/usr/bin/env python3
"""T2: same-seed octave gain interventions and finite-difference power response."""

import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crope.utils import atomic_json, read_prompts, resolution, runtime_environment


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--res", type=resolution, default=(4096, 4096))
    p.add_argument("--method", choices=("base", "pi", "ntk", "ntk2", "yarn", "dype"), default="ntk2")
    p.add_argument("--factor", type=float, default=1.5, help="Logit gain, applied as sqrt(factor) to q and k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--allow-download", action="store_true")
    return p


def octave_edges(grid):
    low = -math.ceil(math.log2(max(grid)))
    return [2.0 ** e for e in range(low, 0)]


def latent_powers(record, edges):
    import numpy as np
    powers = []
    for axis in record["final_latent_axis_power"]:
        f = np.asarray(axis["frequency"])
        p = np.asarray(axis["power"])
        powers.append([float(p[(f >= lo) & (f < hi)].sum()) if ((f >= lo) & (f < hi)).any() else np.nan
                       for lo, hi in zip(edges[:-1], edges[1:])])
    return np.asarray(powers)


def image_powers(path, edges, device):
    """Axis-marginal luminance power, using f_pixel = f_token/16."""
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255
    rgb = torch.from_numpy(rgb).to(device)
    # Decode sRGB before measuring physical luminance spectra.
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = linear @ torch.tensor([0.2126, 0.7152, 0.0722], device=device)
    p = torch.fft.fft2(y - y.mean()).abs().square()
    result = []
    for length, energy in ((y.shape[0], p.sum(1)), (y.shape[1], p.sum(0))):
        freq = torch.fft.fftfreq(length, device=device).abs() * 16
        result.append([float(energy[(freq >= lo) & (freq < hi)].sum())
                       if ((freq >= lo) & (freq < hi)).any() else np.nan
                       for lo, hi in zip(edges[:-1], edges[1:])])
    return np.asarray(result)


def render_heatmap(matrix, output_edges, input_indices, out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from crope._figure_style import DIVERGING, TEXT, figure, save, use_style
    use_style()
    fig, ax = figure(width=TEXT, ratio=0.85)
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    artist = ax.imshow(matrix, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto",
                       interpolation="nearest")
    labels = [f"{1 / hi:g}-{1 / lo:g}" for lo, hi in zip(output_edges[:-1], output_edges[1:])]
    ax.set_xticks(range(len(input_indices)), [labels[i] for i in input_indices], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Perturbed wavelength (tokens)")
    ax.set_ylabel("Measured wavelength (tokens)")
    ax.grid(False)
    fig.colorbar(artist, ax=ax, label=r"$\Delta\log P\,/\,\Delta\log g$", fraction=0.046, pad=0.04)
    save(fig, str(Path(out) / "mtf_heatmap"), formats=("png", "pdf", "svg"))


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    if args.factor <= 0 or args.factor == 1:
        raise ValueError("Factor must be positive and different from one")
    rows = read_prompts(args.prompts)
    if args.count < 1 or args.start < 0 or args.start + args.count > len(rows):
        raise ValueError("Probe range must fit in the prompt file")
    rows = rows[args.start:args.start + args.count]
    import numpy as np
    from crope.runtime import FluxExperiment, load_pipeline
    pipe = load_pipeline(args.device, args.cpu_offload, args.allow_download)
    experiment = FluxExperiment(pipe, args.method, args.res)
    edges = octave_edges(experiment.grid)
    active = []
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if any(((experiment.rope.frequencies(a) / (2 * math.pi) >= lo) &
                (experiment.rope.frequencies(a) / (2 * math.pi) < hi)).any() for a in (1, 2)):
            active.append(b)
    if not active:
        raise ValueError("No rotary dimensions fall within measurable octave bins")
    records = []
    try:
        for row in rows:
            experiment.controller.probe_band = None
            baseline = experiment.sample(row, args.out / "baseline", args.seed, args.steps, args.guidance,
                                         record_power=True)
            base = {"latent": latent_powers(baseline, edges),
                    "image": image_powers(baseline["image"], edges, args.device)}
            for b in active:
                experiment.controller.probe_band = (edges[b], edges[b + 1])
                experiment.controller.probe_factor = args.factor
                result = experiment.sample(row, args.out / f"band_{b:02d}", args.seed, args.steps,
                                           args.guidance, record_power=True)
                perturbed = {"latent": latent_powers(result, edges),
                             "image": image_powers(result["image"], edges, args.device)}
                for domain in ("latent", "image"):
                    delta = (np.log(np.maximum(perturbed[domain], 1e-30)) -
                             np.log(np.maximum(base[domain], 1e-30))) / math.log(args.factor)
                    for axis in range(2):
                        for o in range(len(edges) - 1):
                            records.append({"prompt_id": row["id"], "seed": args.seed, "domain": domain,
                                            "axis": ("height", "width")[axis], "input_bin": b,
                                            "output_bin": o, "input_low": edges[b], "input_high": edges[b + 1],
                                            "output_low": edges[o], "output_high": edges[o + 1],
                                            "gain_factor": args.factor,
                                            "response": float(delta[axis, o]) if np.isfinite(delta[axis, o]) else None})
    finally:
        experiment.close()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "mtf.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    matrices = {}
    for domain in ("latent", "image"):
        matrices[domain] = np.asarray([[np.mean([r["response"] for r in records if r["domain"] == domain
                                                and r["input_bin"] == i and r["output_bin"] == o
                                                and r["response"] is not None])
                                        for i in active] for o in range(len(edges) - 1)])
    diagonals = {name: [float(matrix[b, j]) for j, b in enumerate(active)]
                 for name, matrix in matrices.items()}
    atomic_json(args.out / "mtf.json", {"method": args.method, "res": list(args.res), "factor": args.factor,
                "n_prompts": len(rows), "seed": args.seed, "input_bins": active, "frequency_edges": edges,
                "matrices": {name: m.tolist() for name, m in matrices.items()}, "diagonal": diagonals,
                "all_diagonals_positive": {name: all(x > 0 for x in d) for name, d in diagonals.items()},
                "estimator": "same-seed forward finite difference; mean over prompts and both axes",
                "units": "cycles/token; decoded luminance uses cycles/pixel times16"})
    render_heatmap(matrices["image"], edges, active, args.out)
    (args.out / "mtf_heatmap_caption.txt").write_text(
        f"Decoded luminance response to {args.factor:g}x octave logit gains. "
        f"Same-seed forward finite differences, averaged over {len(rows)} prompts and both spatial axes; "
        f"one seed ({args.seed}), no uncertainty interval. Latent responses are in mtf.csv.\n")
    print(f"T2 decoded-image diagonal: {diagonals['image']}", flush=True)


if __name__ == "__main__":
    main()
