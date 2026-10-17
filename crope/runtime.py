"""Batch-one FLUX inference with step-local state and atomic result records."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from .gain import GainController, install_gain
from .spectral import axis_power, token_grid
from .utils import HF_HOME, atomic_json, file_sha256, image_seed, image_stem

METHODS = ("base", "pi", "ntk", "ntk2", "yarn", "dype", "yarn_temp", "sega", "sega_official",
           "sega_static", "sega_official_static", "constancy", "temp_only", "constancy_nogate", "constancy_sref",
           "constancy_tm", "constancy_tm_nogate", "alias_temp", "alias_temp_sref", "sega_ctm",
           "sega_rms_uniform", "sega_rms_normalized", "sega_ctm_lag")


def load_pipeline(device="cuda", cpu_offload=False, allow_download=False):
    from diffusers import FluxPipeline
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full FLUX inference. Use pytest for tiny CPU validation.")
    if not device.startswith("cuda") and os.environ.get("CROPE_ALLOW_CPU") != "1":
        raise RuntimeError("Full FLUX inference on this shared CPU node is disabled (set CROPE_ALLOW_CPU=1 for a plumbing check)")
    torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    source = "black-forest-labs/FLUX.1-dev"
    if not allow_download:
        # The cache holds only the diffusers-format subfolders (root single-file weights were skipped), which
        # huggingface_hub treats as an incomplete snapshot. Load the snapshot directory directly instead.
        hub = Path(os.environ.get("HF_HOME", "/scratch/09032/saini_2/tmp/hf")) / "hub/models--black-forest-labs--FLUX.1-dev"
        source = str(hub / "snapshots" / (hub / "refs/main").read_text().strip())
    kwargs = {}
    krea = os.environ.get("CROPE_TRANSFORMER")
    if krea:
        # FLUX.1-dev breaks into a dot/halftone lattice at 4096^2 even under the official DyPE code; DyPE and SEGA
        # both evaluate FLUX.1-Krea-dev. Krea ships its own transformer; text encoders and VAE match FLUX.1-dev.
        from diffusers import FluxTransformer2DModel
        kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(krea, subfolder="transformer",
                                                                       torch_dtype=torch.bfloat16)
    pipe = FluxPipeline.from_pretrained(source, torch_dtype=torch.bfloat16, local_files_only=not allow_download, **kwargs)
    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


def code_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def model_revision():
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache("black-forest-labs/FLUX.1-dev", "model_index.json", cache_dir=f"{HF_HOME}/hub")
    if isinstance(path, str):
        parts = Path(path).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    return None


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = [root / "crope" / name for name in
             ("__init__.py", "runtime.py", "gain.py", "rope.py", "sega_official.py", "spectral.py", "utils.py")]
    paths += [root / "scripts" / name for name in ("sample.py", "calibrate.py", "probe_mtf.py")]
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class FluxExperiment:
    def __init__(self, pipe, method, resolution, profile=None, static_profile=None, eta=None,
                 target="profile", temperature=None, probe_band=None, probe_factor=1.5, gate=None):
        from .rope import FluxRoPE
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}")
        self.pipe, self.method, self.resolution = pipe, method, tuple(resolution)
        self.grid = tuple(n // 16 for n in resolution)
        # mu = max_shift at every resolution (see sample()); set before any config is recorded.
        if hasattr(self.pipe.scheduler, "register_to_config"):
            self.pipe.scheduler.register_to_config(max_image_seq_len=max(4096, self.grid[0] * self.grid[1]))
        self.axes_dims = tuple(pipe.transformer.config.axes_dims_rope)
        rope_method = method if method in ("base", "pi", "ntk", "ntk2", "yarn", "dype", "sega_official") else "ntk2"
        if method in ("sega_official_static", "alias_temp", "alias_temp_sref", "sega_ctm", "sega_ctm_lag",
                      "sega_rms_uniform", "sega_rms_normalized"):
            rope_method = "sega_official"
        # Controlled comparison: run any gain controller on another method's RoPE base (recorded in config["rope"]).
        rope_method = os.environ.get("CROPE_ROPE", rope_method)
        mode = "yarn_temp" if method == "yarn" else (method if method in GainController.MODES else "none")
        self.rope = FluxRoPE(axes_dims=self.axes_dims, theta=10000, method=rope_method, grid=self.grid)
        self.controller = GainController(mode, self.grid, self.axes_dims, self.rope, profile, static_profile,
                                         eta, target, temperature, probe_band, probe_factor, gate)
        if self.controller.time_matched:
            self.method = self.controller.mode
        self.original_rope = pipe.transformer.pos_embed
        self.original_processors = dict(pipe.transformer.attn_processors)
        pipe.transformer.pos_embed = self.rope
        install_gain(pipe.transformer, self.controller)
        self.handles = [pipe.transformer.register_forward_pre_hook(self._before, with_kwargs=True),
                        pipe.transformer.register_forward_hook(self._after, with_kwargs=True)]
        self.official = None
        if method in ("sega_official", "sega_ctm", "sega_ctm_lag", "sega_rms_uniform", "sega_rms_normalized"):
            from .sega_official import OfficialSEGA
            self.official = OfficialSEGA(grid=self.grid, axes_dims=self.axes_dims, theta=10000)
            self.handles.append(pipe.transformer.x_embedder.register_forward_hook(self._projected))
        self.profile_path, self.static_path = profile, static_profile
        self.profile_hash, self.static_hash = file_sha256(profile), file_sha256(static_profile)
        self.revision = code_revision()
        self.implementation_hash = implementation_hash()
        self.model_revision = model_revision()
        self.reset()

    def reset(self, observer=None):
        self.step = 0
        self.previous_x0 = None
        self.previous_x0_sigma = None
        self.current_xt = None
        self.observer = observer
        self.controller.reset()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.pipe.transformer.pos_embed = self.original_rope
        self.pipe.transformer.set_attn_processor(self.original_processors.copy())

    def _before(self, module, args, kwargs):
        packed = kwargs.get("hidden_states", args[0] if args else None)
        self.current_xt = token_grid(packed.detach(), self.grid)
        # Pipeline timesteps are cast to bf16; the scheduler retains the precise sigma.
        if self.step >= len(self.pipe.scheduler.sigmas) - 1:
            raise RuntimeError("Multiple transformer calls per step are unsupported; true CFG must remain off")
        self.sigma = float(self.pipe.scheduler.sigmas[self.step])
        timestep = kwargs.get("timestep")
        self.model_timestep = float(timestep.flatten()[0]) if timestep is not None else None
        self.rope.set_timestep(self.sigma)
        if self.official is None:
            self.controller.update(self.current_xt, self.sigma, self.previous_x0)
            self.controller.history[-1]["model_timestep"] = self.model_timestep

    def _projected(self, module, args, projected):
        m = self.official.update(projected.detach(), self.sigma)
        if self.method in ("sega_ctm_lag", "sega_rms_uniform", "sega_rms_normalized"):
            self.controller.update(self.current_xt, self.sigma, self.previous_x0, m,
                                   x0_sigma=self.previous_x0_sigma)
            self.controller.history[-1].update(model_timestep=self.model_timestep,
                                               official=self.official.diagnostics)
            return
        if self.method == "sega_ctm":
            # SEGA supplies its complete q/k gain. update adds only the time-matched
            # logit correction eta*w*clip(e,-1,1), hence a half exponent on q and k.
            self.controller.update(self.current_xt, self.sigma, self.previous_x0, m)
            self.controller.history[-1].update(model_timestep=self.model_timestep,
                                               official=self.official.diagnostics)
            return
        self.controller.qk_scale = m
        self.controller.log_g = 2 * m.abs().clamp_min(1e-30).log()
        info = {"step": self.step, "sigma": self.sigma, "model_timestep": self.model_timestep,
                "mode": "sega_official",
                "official": self.official.diagnostics}
        self.controller._log(info)

    def _after(self, module, args, kwargs, output):
        v = output[0] if isinstance(output, tuple) else output.sample
        self.previous_x0 = self.current_xt.float() - self.sigma * token_grid(v.detach().float(), self.grid)
        self.previous_x0_sigma = self.sigma
        if self.observer is not None:
            self.observer(self.step, self.sigma, self.current_xt, self.previous_x0)
        self.current_xt = None
        self.step += 1

    def config(self, row, seed, steps, guidance):
        import diffusers
        return {"schema_version": 1, "method": self.method, "res": f"{self.resolution[0]}x{self.resolution[1]}",
                "prompt_id": str(row["id"]), "prompt": row["prompt"], "seed": int(seed),
                "derived_seed": image_seed(row["id"], seed), "steps": steps, "guidance": guidance,
                "model": "black-forest-labs/FLUX.1-dev", "model_revision": self.model_revision,
                "dtype": "bfloat16", "batch_size": 1, "attention": "pytorch_sdpa_flash",
                "diffusers": diffusers.__version__, "torch": torch.__version__, "code_revision": self.revision,
                "implementation_sha256": self.implementation_hash,
                "grid": list(self.grid), "eta": self.controller.eta, "target": self.controller.target,
                "temperature": self.controller.temperature, "profile_sha256": self.profile_hash,
                "static_profile_sha256": self.static_hash, "rope": self.rope.metadata(),
                "probe_band": self.controller.probe_band, "probe_factor": self.controller.probe_factor,
                "scale_choice": "spatial axes per-side; scalar temperature geometric-mean side",
                "side_scale": [n / 64 for n in self.grid], "area_scale": self.grid[0] * self.grid[1] / 4096,
                "sega_sref_area_alternative": (self.grid[0] * self.grid[1] / 4096) ** 0.08,
                "scheduler": dict(self.pipe.scheduler.config)}

    @torch.inference_mode()
    def generate(self, row, seed=0, steps=28, guidance=3.5, output_type="pil", observer=None):
        self.reset(observer)
        h, w = self.resolution
        if h * w >= 2048 * 4096:
            self.pipe.vae.enable_tiling()
        generator = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], seed))
        cuda = torch.cuda.is_available() and str(self.pipe._execution_device).startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        # Official DyPE and SEGA pipelines pass max_seq_len = image_seq_len, i.e. mu = max_shift (1.15) at any
        # resolution. The diffusers default extrapolates mu linearly (~11.5 at 4096^2), collapsing all sigmas to ~1.
        self.pipe.scheduler.register_to_config(max_image_seq_len=max(4096, (h // 16) * (w // 16)))
        start = time.perf_counter()
        result = self.pipe(prompt=row["prompt"], height=h, width=w, num_inference_steps=steps,
                           guidance_scale=guidance, generator=generator, max_sequence_length=512,
                           output_type=output_type)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if self.step != steps:
            raise RuntimeError(f"Expected {steps} gain updates, observed {self.step}")
        if self.controller.history[-1]["sigma"] > 0.9:
            print("Schedule diagnostic: the unchanged default shift has no low-sigma refinement phase "
                  "at this resolution. Inspect scripts/diagnose_schedule.py before a campaign.", flush=True)
        return result.images[0], elapsed, torch.cuda.max_memory_allocated() if cuda else None

    def sample(self, row, out, seed=0, steps=28, guidance=3.5, record_power=False):
        from PIL import Image
        out = Path(out)
        stem = image_stem(row["id"], seed)
        png, metadata = out / f"{stem}.png", out / f"{stem}.json"
        config = self.config(row, seed, steps, guidance)
        digest = fingerprint(config)
        if png.exists() and metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved.get("complete"):
                def _norm(c):
                    # The first image of a process used to record max_image_seq_len before it was set; sigmas
                    # were identical (mu = max_shift either way), so ignore this bookkeeping field.
                    c = json.loads(json.dumps(c)); sch = c.get("scheduler", {})
                    for k in ("max_image_seq_len", "_use_default_values"):
                        sch.pop(k, None)
                    return fingerprint(c)
                if saved.get("config_fingerprint") != digest and _norm(saved.get("config", {})) != _norm(config):
                    raise ValueError(f"Existing result has different settings: {metadata}. Choose a new --out.")
                with Image.open(png) as im:
                    if im.size != self.resolution[::-1]:
                        raise ValueError(f"Existing image has incorrect dimensions: {png}")
                    im.verify()
                if file_sha256(png) != saved["png_sha256"]:
                    raise ValueError(f"Existing PNG checksum differs: {png}")
                if record_power and "final_latent_axis_power" not in saved:
                    raise ValueError("Probe output lacks latent spectrum; choose a new output directory")
                print(f"skip {row['id']} {self.method} {config['res']}", flush=True)
                return saved
        image, seconds, peak_memory = self.generate(row, seed, steps, guidance)
        out.mkdir(parents=True, exist_ok=True)
        temporary = png.with_name(png.name + f".tmp.{os.getpid()}")
        image.save(temporary, format="PNG")
        os.replace(temporary, png)
        record = {"complete": True, "config": config, "config_fingerprint": digest,
                  "image": str(png.resolve()), "png_sha256": file_sha256(png), "seconds": seconds,
                  "seconds_scope": "prompt encoding, denoising, and VAE decode; excludes model loading and PNG I/O",
                  "peak_memory_bytes": peak_memory, "gains": self.controller.history,
                  "sigmas": [r["sigma"] for r in self.controller.history]}
        if record_power:
            record["final_latent_axis_power"] = [
                {"lower": a.lower.cpu().tolist(), "upper": a.upper.cpu().tolist(),
                 "frequency": a.frequency.cpu().tolist(), "power": a.power.cpu().tolist(),
                 "count": a.count.cpu().tolist()}
                for a in axis_power(self.previous_x0)]
        atomic_json(metadata, record)
        print(f"saved {row['id']} {self.method} {config['res']} seconds={seconds:.3f}", flush=True)
        return record
