# Implementation evidence

This task builds code and runs CPU checks only. No scheduler operations, SSH,
model inference, environment installs, or task enqueueing have been performed.
The code and output roots are the two `constancy_rope` directories.

Completed units:

- Full design and both supplied CPU references read.
- Official DyPE, SEGA, and UltraImage Flux branches shallow-cloned under scratch.
- RoPE tests: 28 passed with mend Python, diffusers 0.39.0. Base, positive-sigma
  DyPE, and released SEGA numerical parity include exact checks.
- Gain controller, step hooks, sampler, native/static calibration, T2 probe,
  evaluation adapters/statistics, prompt export, and staged task tooling complete.
- Final CPU suite: 107 passed, one external TRANSFORMERS_CACHE deprecation warning,
  15.14 seconds, using mend Python with torch 2.11.0+cu128 and diffusers 0.39.0.
- Tests cover all required numerical invariants, 3072-grid SEGA binning,
  calibration and sampling recovery, bootstrap pairing, task dependencies,
  all seven CLI help paths, schedule diagnostics, and PNG/PDF/SVG export.
- Heatmap renderer fixture passed the figure audit and was inspected. It is
  explicitly marked synthetic test data, not a generation or T2 measurement.
- All 195 cached 4K evaluation prompts and references exported.
- All-stage dry-run planned 347 scripts; real task queue had 0 files before and after.
- Local commit attempt failed: `.git/index.lock`: Read-only file system.

Decisions:

- The explicit reference-parity requirement takes precedence for the name
  `sega`. Official behavior is also implemented as `sega_official` and tested.
- YaRN follows the design's explicit wavelength-ratio ramp, despite its
  difference from standard rotation-count YaRN. DyPE follows upstream exactly.
- Gain methods use NTK2 frequencies. Standalone PI, NTK, YaRN, and DyPE retain
  their specified frequencies, overriding the older STATE shorthand.
- Rectangular scalar temperatures use the geometric-mean side scale; spatial
  frequency scaling stays per-axis. Official SEGA uses its longest-side scale.
- T2 estimates a finite difference for a 1.5x logit gain; q/k scales are its
  square root. Both latent and decoded-image power responses are reported.
- Task stages are explicit dependencies because sorted prefixes cannot impose
  completion ordering across fleet workers. T1 SEGA-static runs follow a
  completed target-resolution SEGA calibration, without sleeping workers.
- Source hashes enter sample fingerprints because a commit alone cannot
  identify a dirty working tree.
- Flash-only attention is enforced inside FLUX blocks; short masked CLIP/T5
  attention retains its own compatible backend selection.
- Float64 FFT frequency coordinates preserve the reference's integer-truncation
  behavior at non-power-of-two grids while spectra may remain float32.

Schedule finding, verified with the cached FLUX scheduler configuration:

| Resolution | mu | Last evaluated sigma | Distinct bf16 model timesteps |
| --- | --- | --- | --- |
| 1024x1024 | 1.15 | 0.104720898 | 28 |
| 2048x4096 | 6.003333333 | 0.937467515 | 8 |
| 3072x3072 | 6.696666667 | 0.967730463 | 6 |
| 4096x4096 | 11.55 | 0.999739885 | 1 |

All high-resolution schedules above have zero evaluated steps below sigma 0.3.
Official DyPE/SEGA fix mu to 1.15. The design explicitly requests the unchanged
default schedule, which is preserved. This is a material obstacle to interpreting
the high-resolution comparisons, not an observed image-quality result.

Receipts on project scratch: `validation/cpu_tests.log`, `cpu_tests.xml`,
`task_plan.json`, `task_check.json`; schedule evidence: `diagnostics/default_schedule.json`.
At the last check, crope still lacked diffusers, ImageReward, torch-fidelity,
and torchmetrics; the environment builder remains external to this task.
HPSv3 is optional and does not trigger a download of its unlisted large base model.
Open validation: real GPU smoke, GH200 memory, calibrated throughput, actual T2,
and learned metric model loading/scoring. No research efficacy claims are made.
