# Diagnostic preparation, 2026-10-19 UTC

Prepared before any new generation. These are implementation and input-pipeline
checks, not image-quality results. No task was published by this agent.

## P2: checkpoint by guidance, official DyPE

`scripts/strengthen_diagnostics.py` imports the official `flux.pipeline_flux`
and `flux.transformer_flux` directly from the preserved DyPE checkout at
`/scratch/09032/saini_2/constancy_rope/third_party/DyPE`.
The wrapper requires clean commit `436fe612fb7695fa2d5dc32d63390c048ba6033c`;
all official flux Python source hashes, wrapper hash, versions, settings, prompt
hash and image SHA256 are saved in sidecars. Neither official source nor project
generation runtime is changed by this diagnostic. Imports were checked on CPU
in the existing crope uv environment without loading weights.

Use dev manifest rows 0 through 7 inclusive, both transformer checkpoints and
both guidance values 3.5 and 4.5. Each image uses the original diagnostic's
CUDA generator reset to seed 42, 28 steps, 4096x4096, BF16 and tiled VAE. This
is deliberately the prior official diagnostic RNG policy, rather than the
prompt-hashed CPU seeds used by the project's main generation script. Noise is
matched across all four matrix cells for each prompt. All four cells use the
FLUX.1-dev text encoders, VAE and scheduler; only the transformer checkpoint
changes. Source versions and fixed snapshots are recorded:

- FLUX.1-dev: `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`.
- FLUX.1-Krea-dev: `8162a9c7b05a641be098422bf2fcf335615c2f28`.
- DyPE positional mode `yarn`, `dype=True`, max text length 512.
- Its exact released yarn rule multiplies beta thresholds `(1.25, 0.75)` and
  gamma thresholds `(16, 2)` by `timestep**2`, with coefficient 1. There are
  no `dype_scale` or `dype_exponent` constructor arguments. A coefficient 2
  appears only in the unused `ntk` branch. The original amendment's
  `dype_scale2` wording needs correction before P2 publication.
- Official pipeline passes `image_seq_len` as the upper sequence anchor in
  `calculate_shift`, so mu is 1.15 at 4K. Scheduler config and actual timestep
  and sigma arrays are saved after each image.

Suggested eight spool entries: four cells, dev rows 0:4 and 4:8 per cell. Prefix
`76_`, with generation invocations of the following form after prereg amendment:

```bash
python scripts/strengthen_diagnostics.py --checkpoint dev --guidance 3.5 --start 0 --end 4
```

Replace checkpoint with `dev` or `krea`, guidance with `3.5` or `4.5`, and row
range with `(0,4)` or `(4,8)`. Standard worker environment, telemetry, eight-thread
caps and offline W&B when credentials are absent apply. Wrapper output defaults
to `scratch/diag/ckpt_matrix/<method>/4096x4096/` and uses unique evaluator methods:
`official_dype_dev_g3p5`, `official_dype_dev_g4p5`,
`official_dype_krea_g3p5`, `official_dype_krea_g4p5`.

Use the existing eval and ImageReward scripts on `scratch/diag/ckpt_matrix` with
the existing eval manifest. There are 32 intended outputs, exactly eight per
method. These are development diagnostics. Preserve guidance-specific results;
do not call pooled checkpoint differences a guidance-independent effect.

Resume verifies complete config identity, PNG hash and dimensions. Existing
partial or different samples are preserved and rejected, requiring a deliberate
move aside before repair. The wrapper logs each completed image to W&B. Twelve
CPU tests passed for cell identity, fixed inputs, fixed seed, source pin and
resume corruption detection. GPU execution remains untested until fleet work.

## P7: exact MUSIQ input audit

The installed package is pyiqa 0.1.16. The project calls
`pyiqa.create_metric('musiq')`, which selects the KonIQ10k model. The project
passes the complete RGB image as float32 in [0,1] without spatial resize
(`crope/metrics/models.py`, `tensor_rgb` and `PyiqaScorer.score`). In eval mode,
MUSIQ converts this tensor to [-1,1] and constructs three scales:

| Input | 224 scale | 384 scale | Native scale | Total patches |
|---|---:|---:|---:|---:|
| 4096x4096 image | 49 | 144 | 16,384 | 16,577 |
| 1024x1024 crop | 49 | 144 | 1,024 | 1,217 |

Both resized branches preserve aspect ratio and use PyTorch bicubic interpolation
with `align_corners=False` and the default `antialias=False`. The preprocessing
docstring says Gaussian, but the executable call is bicubic. Each branch uses
nonoverlapping 32x32 patches with stride 32 and TensorFlow-style SAME padding
when necessary. Neither square size in this experiment needs native padding.
The native cap is `max_seq_len_from_original_res=-1`, meaning all native patches
are retained. There is no native crop, subsampling or maximum patch count. Spatial
position indices hash to a 10x10 grid; that is an embedding lookup, not image
downsampling. A class token is added after patch embedding, giving 16,578
transformer tokens for the full 4K image.

We executed only `get_multiscale_patches` on CPU tensors of both sizes, without
model weights, and observed the counts above and all-valid masks. Source SHA256,
package versions and tensor shapes are in `diagnostics_musiq_schedule_audit.json`.
The project separately scores four fixed random 1024x1024 crops per image, shared
across methods, and averages their MUSIQ outputs. Thus full-image MUSIQ really
does include every native-scale pixel, but also incorporates two resized views.
It should be called a multiscale full-image quality score, not a direct sharpness
measurement.

Suggested paper sentence:

> We use pyiqa 0.1.16 MUSIQ (KonIQ10k), which processes 32x32 patches from the
> native image and bicubic 224- and 384-pixel views; its native branch retains
> all 16,384 patches at 4096x4096 without a patch cap. Crop MUSIQ averages four
> matched random 1024x1024 crops, each processed through the same three scales.

Executable source locations, relative to crope's site-packages:
`pyiqa/default_model_configs.py:279`, `pyiqa/archs/musiq_arch.py:318`,
`pyiqa/archs/musiq_arch.py:396`, `pyiqa/data/multiscale_trans_util.py:39`,
`pyiqa/data/multiscale_trans_util.py:104` and
`pyiqa/data/multiscale_trans_util.py:157`.

## P7: default flow shift

The CPU scheduler reproduction uses the cached FLUX.1-dev scheduler and exactly
the same 28-step construction, timestep cast and division order as FluxPipeline.
It loads no weights. Full arrays and configuration are archived in the audit JSON.

| Schedule | mu | Last evaluated sigma | Distinct BF16 model timesteps | Evals sigma < 0.3 | Largest Euler step |
|---|---:|---:|---:|---:|---:|
| Diffusers default upper anchor 4096 | 11.55 | 0.9997398853 | 1 | 0 | 0.9997398853 |
| Official upper anchor 65536 | 1.15 | 0.1047208980 | 28 | 3 | 0.1047208980 |

Default shifts are almost one, not mathematically equal to one. The BF16 model
timesteps collapse to one unique value; the scheduler still uses distinct FP32
sigmas and ends with a very large Euler step. Numeric collapse alone establishes
no image-quality effect. The requested eight default-shift images will measure
that consequence descriptively.

Recommended runtime hook, owned by the main agent: explicit
`schedule_mode=official|diffusers_default`, recorded in every config. The default
mode must leave `max_image_seq_len=4096` unchanged at BOTH existing scheduler
mutation sites (FluxExperiment initialization and generate). Use existing dev
rows 0:8, `sega_official`, seed 0, guidance 4.5, 28 steps, with method identity
`sega_official_default_shift`. This creates direct pairs against the existing
official-schedule pilot images. Prefix `78_`; keep the comparison descriptive.
