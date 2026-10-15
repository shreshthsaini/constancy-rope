# Constancy RoPE

Resumable FLUX.1-dev comparisons at 4096x4096, 3072x3072, and rectangular resolutions.
The default protocol is bf16, batch 1, 28 steps, guidance 3.5, the pipeline's shifted
flow scheduler, PyTorch SDPA flash attention, and tiled large-image VAE decode.
CPU tests use tiny random models. Full inference requires a GPU.

The requested default scheduler has a serious 4K limitation: its last evaluated
sigma is about 0.99974, followed by a large step to zero. All 28 bf16 model
timesteps coincide at that resolution. This implementation preserves the spec.
Run `scripts/diagnose_schedule.py` on CPU before interpreting a GPU smoke;
upstream DyPE and SEGA use a different shift setting.

Use the existing `crope` uv environment. Scripts import this checkout directly.
Core and optional evaluation dependencies are declared in `pyproject.toml`; the
concurrent environment build owns `scripts/requirements.lock.txt`. Do not install
into `mend`. While `crope` is incomplete, `mend/bin/python` can run CPU tests read-only.

```bash
cd /work/09032/saini_2/vista/Projects/Exploration/constancy_rope
PY=/work/09032/saini_2/vista/library/uv-envs/crope/bin/python
CR=/scratch/09032/saini_2/constancy_rope
export HF_HOME=/scratch/09032/saini_2/tmp/hf
export TORCH_HOME=/scratch/09032/saini_2/tmp/cache
export XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache
export WANDB_DIR=$CR/wandb WANDB_PROJECT=constancy-rope WANDB_MODE=offline
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
```

Export the 195 4K eval captions and matching cached reference paths. Omit
`--offline` to fetch missing files from the specified dataset only.

```bash
"$PY" scripts/prompts.py --offline --out "$CR/prompts/aesthetic4k_eval.jsonl" \
  --references "$CR/prompts/aesthetic4k_refs.jsonl"
PROMPTS=$CR/prompts/aesthetic4k_eval.jsonl
REFS=$CR/prompts/aesthetic4k_refs.jsonl
```

Calibrate the native profile before Constancy sampling. Each prompt has a
resumable trace containing every step's axis log power. The final profile retains
the mean over steps with sigma<0.3 and the mean over prompts at each native step.
Use separate calibration captions or reserve these IDs from
evaluation; the following examples reserve the first 16 rows.

```bash
"$PY" scripts/calibrate.py --mode native --prompts "$PROMPTS" --count 16 \
  --out "$CR/calib/flux_dev_profile.npz"
"$PY" scripts/sample.py --method constancy --res 4096x4096 --prompts "$PROMPTS" \
  --start 16 --end 17 --seed 0 --profile "$CR/calib/flux_dev_profile.npz" \
  --out "$CR/out/main/constancy/4096x4096"
```

Supported methods: `base`, `pi`, `ntk`, `ntk2`, `yarn`, `dype`, `sega`,
`sega_static`, `constancy`, `temp_only`, `constancy_nogate`, `constancy_sref`,
`constancy_tm`, `constancy_tm_nogate`, `yarn_temp`, `sega_official`, and
`sega_official_static`. Constancy accepts `--eta`,
`--target profile|slope`, and `--temperature logN|sref|none`.
Seeds depend only on prompt ID and seed, across methods and chunk boundaries.
Each PNG has a JSON sidecar with settings, source/model hashes, time, peak memory,
and per-step logit gains by dimension and axis. Completed matching results are
skipped after integrity checks; conflicting settings require a new output path.
Model loading defaults to the existing cache. `--cpu-offload` is an explicit
fallback for VRAM pressure; attention slicing is never enabled.

`constancy_tm` matches the native log band density at the current scheduler sigma,
using the previous step's x0 estimate. It interpolates linearly in sigma, clamps
outside the native schedule, and uses Constancy's log-frequency interpolation and
low-frequency extrapolation. Its 16..64-token anchor bands do not require a
visibility threshold; an empty anchor set gives zero offset. The Wiener gate,
pair mapping, temperature, axis-0 behavior, and logs follow Constancy. The first
step applies temperature only. `--gate off` selects `constancy_tm_nogate`; enqueue
the ablation by that method name. Both modes require `--target profile` and a
profile containing `sigma_steps`, `log_power_t_h`, and `log_power_t_w`.

Rebuild both profiles from saved traces on CPU without loading a model or changing
the inputs. The late-step reduction keeps its original source order when the
sibling profile records the same trace set.

```bash
"$PY" scripts/calibrate.py --mode native-rebuild \
  --traces "$CR/calib/krea_profile_traces" --out "$CR/calib/krea_profile_tm.npz"
"$PY" scripts/sample.py --method constancy_tm --res 4096x4096 --prompts "$PROMPTS" \
  --profile "$CR/calib/krea_profile_tm.npz" --eta 1 \
  --out "$CR/out/main/constancy_tm/4096x4096"
```

SEGA-static calibration consumes completed `sega` logs at the target resolution
and stores the arithmetic mean of its per-axis modulators. It performs no FFT
during subsequent sampling.

```bash
"$PY" scripts/calibrate.py --mode sega-static --runs "$CR/out/pilot/sega/4096x4096" \
  --res 4096x4096 --out "$CR/calib/sega_static_4096x4096.npz"
"$PY" scripts/sample.py --method sega_static --res 4096x4096 --prompts "$PROMPTS" \
  --start 16 --end 24 --sega-static-profile "$CR/calib/sega_static_4096x4096.npz" \
  --out "$CR/out/pilot/sega_static/4096x4096"
"$PY" scripts/probe_mtf.py --prompts "$PROMPTS" --start 16 --count 1 \
  --res 4096x4096 --out "$CR/probe/4096x4096"
```

T2 uses same-seed 1.5x octave logit-gain interventions throughout sampling. It writes
latent and decoded-image finite differences to CSV/JSON, plus PNG/PDF/SVG heatmaps.
T3 uses the NTK2 pilot's per-octave spectra; T1 compares SEGA-static with SEGA.

```bash
"$PY" scripts/eval.py --root "$CR/out/main" --prompts "$PROMPTS" --references "$REFS" \
  --reference-method sega --out "$CR/eval/main/4096x4096" --res 4096x4096
"$PY" scripts/enqueue.py --stage all --dry-run --prompts "$PROMPTS" --references "$REFS" \
  --start 16 --end 116 --show-scripts
"$PY" -m pytest -q
"$PY" scripts/diagnose_schedule.py --out "$CR/diagnostics/default_schedule.json"
```

Evaluation writes per-image scores, summary CSV/JSON, paired deltas, 95% prompt
bootstrap intervals, and W&B records. Crops and bootstrap draws use fixed seeds.
FID/KID use canonical Inception weights, with whole images resized internally
to 299 and random 1024 crops. CLIP is 100*max(cosine,0); PickScore is the raw scaled
similarity. MUSIQ and CLIP-IQA receive full images and 1024 crops. HPSv3 is optional
under `--metrics all` if its package or complete Qwen base is absent. Other missing
requested metrics fail explicitly; `--allow-missing` permits documented partial
reports. `--metrics spectrum,tiling --no-wandb` needs no learned weights.

Task stages are `smoke`, `calibration`, `probe`, `pilot`, `pilot-static`, `main`,
and `eval`. Prefixes 00/10/20/30/31/50/90 set priorities. A worker can claim a later
prefix before an earlier task finishes, so generate each real stage only after
its required artifacts exist. `all` is dry-run only. SEGA pilot generation builds
its static profile, then `pilot-static` runs T1. `--seconds-per-image` controls
30-45 minute chunks; the default 300 seconds is an unmeasured planning estimate.
Use the smoke timing to update it. `slurm/worker_loop.sh` is unchanged.

The reference SEGA formula conflicts with released SEGA code, so `sega` matches
the supplied CPU reference and `sega_official` implements the released algorithm.
YaRN follows the design's explicit wavelength-ratio ramp. DyPE matches upstream,
including its spatial amplitude. [THIRD_PARTY.md](crope/THIRD_PARTY.md) records
revisions, licensing, rectangular scaling, and these distinctions. CPU parity
does not establish 4K quality, GH200 memory fit, scorer availability, or speed.
