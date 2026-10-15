# Time-matched Constancy implementation and CPU validation

Completed 2026-10-15. No GPU generations or quality claims.

## Implementation

- Native calibration retains the late-step profile and adds sigma_steps [S],
  log_power_t_h [S,B_h], and log_power_t_w [S,B_w]. Each time row is the mean
  log band density over prompts. Frequency arrays remain unchanged.
- native-rebuild reads existing traces without loading a pipeline. It checks
  normalization, shapes, finite values, aligned sigmas/frequencies, settings,
  and unique prompt/seed pairs. It reuses the sibling profile's source order
  when that profile records the same trace set, preserving float32 reductions.
- constancy_tm interpolates native log density linearly in sigma and then
  log frequency. Sigma is clamped at the native endpoints. Low-frequency
  extrapolation uses the same two lowest bands as Constancy.
- Anchors cover wavelengths 16..64 tokens without a visibility threshold.
  Empty anchors give c=0 and still compute the deficit. The Wiener gate,
  correction clipping, pair interpolation, first-step behavior, axis-0 gain,
  temperature, and logging follow Constancy.
- sample.py --gate off selects constancy_tm_nogate. Runtime metadata records
  that effective mode. enqueue.py accepts both method names and rejects a
  legacy profile before creating a spool. Both modes require --target profile.
- Existing modes and default method lists are unchanged.

Changed implementation: crope/{gain,runtime,spectral}.py and
scripts/{calibrate,sample,enqueue}.py.
Changed tests: tests/test_{calibration,cli,gain,runtime,tooling}.py;
new tests/test_time_profile.py.
Changed documentation: README.md, HANDOVER.md, .agent/STATE.md, and this note.
Pre-existing changes to .agent/notes/results.md and codex_brief_3.md were preserved.

## Validation

Full suite: 156 passed in 13.22 seconds. This includes zero deficits along a
native trajectory, positive high-band deficits and corrections at sigma=0.6
under additional blur, gate and anchor separation, empty anchors, first-step
and axis-0 behavior, pair mapping at 4096 pixels, previous-x0 timing with exact
scheduler sigmas, two-axis interpolation/clamping/extrapolation, CLI/profile
validation, and unchanged prior-mode regression coverage.

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/constancy_tm_mpl \
CROPE_TEST_OUTPUT=/tmp/constancy_tm_fixtures \
/work/09032/saini_2/vista/library/uv-envs/crope/bin/python -m pytest -q \
  -o cache_dir=/tmp/constancy_tm_pytest_cache --basetemp=/tmp/constancy_tm_tests

OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
PYTHONDONTWRITEBYTECODE=1 \
/work/09032/saini_2/vista/library/uv-envs/crope/bin/python scripts/calibrate.py \
  --mode native-rebuild \
  --traces /scratch/09032/saini_2/constancy_rope/calib/krea_profile_traces \
  --out /scratch/09032/saini_2/constancy_rope/calib/krea_profile_tm.npz
```

Rebuilt profile: 16 prompts, 28 sigma steps, nine bands per axis. All 14 existing
profile fields compare exactly with krea_profile.npz, including the late-step
arrays. The time arrays equal the direct per-step prompt means. All 17 SHA256
hashes for the original profile and input traces are unchanged.

Output SHA256:
`1cd6bceb50dea48607b125681ade330e5203cc8342eeb99834c3493d41b2cc45`.
Machine-readable receipt (input hashes and both recorded/interpolated values):
`/scratch/09032/saini_2/constancy_rope/codex_tmp/constancy_tm_validation.json`.

## Native width-axis log band densities

Frequencies in cycles/token, in the order used below:

```text
0.01858136 0.03716272 0.05255603 0.07432544 0.10511205 0.14865088 0.21022411 0.29730177 0.42044821
```

Nearest saved native steps to requested sigmas 0.9, 0.6, 0.3, and 0.1:

```text
sigma       step  log_power_t_w
0.90453076    7    5.514054 4.709264 4.098067 3.492521 3.001645 2.365833 1.615944 0.947054 0.348102
0.59935671   19    5.413790 4.620918 4.027957 3.478842 3.050471 2.526902 2.038580 1.663478 1.419673
0.27482799   25    5.409727 4.652107 4.107520 3.626928 3.275101 2.870965 2.534554 2.297933 2.159770
0.10472090   27    5.415684 4.668790 4.136714 3.675616 3.342772 2.964196 2.655601 2.442650 2.320257
```

Interpolated controller targets at the exact requested sigmas (0.1 clamps to
the final native step at 0.10472090):

```text
0.9   5.508068 4.705399 4.089170 3.484747 2.999571 2.370983 1.633482 0.977205 0.398649
0.6   5.413757 4.620861 4.027792 3.478564 3.050084 2.526287 2.037637 1.662175 1.418121
0.3   5.408706 4.648442 4.100253 3.614780 3.257383 2.845713 2.500786 2.256800 2.113306
0.1   5.415684 4.668790 4.136714 3.675616 3.342772 2.964196 2.655601 2.442650 2.320257
```

The shared CPU node was used with OMP_NUM_THREADS=8 and fewer than 20 processes.
No installs, commits, SSH, scheduler commands, or scratch deletions were run.
The only calibration-directory write was krea_profile_tm.npz and its atomic
temporary file. Test fixtures and caches were redirected to /tmp. No background
processes remain from this task.
