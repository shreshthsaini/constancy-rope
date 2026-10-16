## Goal
Build and CPU-validate the resumable FLUX resolution-extrapolation codebase in design.md.
Research generations and paper claims follow separate GPU validation.

## Status and next action
2026-10-15 18:15. Krea campaign. CR3 4n until 10-16 11:40. Main baselines (ntk2, dype, sega_official x 195) running; constancy eta=1 main held (taskq/hold).
DEV (prompts 0-7) result: constancy(eta1,2,4) == temp_only (IR ~1.28) < sega_official (IR 1.42; pick 23.41 vs 23.10; clip 30.32 vs 30.10); nogate worse (IR 1.00). Correction inactive until sigma<=0.35 because the target is the LATE native profile.
Codex3 complete: constancy_tm and constancy_tm_nogate; 156 tests passed in 13.22s. CPU native-rebuild created calib/krea_profile_tm.npz (16 prompts, 28 steps, 9 bands/axis). All 14 old profile fields equal exactly; 17 input hashes unchanged. Changes uncommitted; no queue, Slurm, or paper changes. Details: notes/constancy_tm.md.
Sweep evals queued (41_eval_sweep_*, 41_eval_pilot_v3). Sweep outputs moved to out_krea/sweep/, eval roots out_krea/sweeproot_* (symlinks).
NEXT: constancy_tm dev runs done (out_krea/devtm_eta{1,2}); correction now active from step ~8. Scoring: 42_eval_devtm_eta* + eval_krea/ir/devtm_eta*.csv. Compare vs sega_official on dev; if better -> requeue constancy main as constancy_tm (profile calib/krea_profile_tm.npz) and report held-out prompts 8-194.

## Map
| note | read it when |
| notes/design.md | implementing or changing any method, baseline, or metric |
| notes/implementation.md | reviewing validation, deviations, and operational limitations |
| ../crope/THIRD_PARTY.md | inspecting upstream revisions, licensing, and SEGA differences |
| notes/codex.md | reviewing the original implementation dispatch |
| notes/constancy_tm.md | reviewing time-matched controller changes, tests, rebuilt profile, or width-axis values |

## Decisions
- Model is FLUX.1-dev for every method, as requested.
- Gain methods use NTK2; standalone PI/NTK/YaRN/DyPE follow their explicit design rules.
- sega follows the supplied reference; sega_official follows released projected-feature logic because they differ.
- Task stages require completed prerequisites because file prefixes alone cannot order a worker fleet.
- Batch one, bf16, native scheduler shift, SDPA flash, and tiled large-image VAE decode preserve the protocol.

- Main SEGA baseline = sega_official (released code, github rajabi2001/sega). Why: released code differs from paper (alloc 1 - 0.15*spread*tanh(1.5z), clipped >= 1, x_embedder features); paper-equation 'sega' kept as an ablation. The earlier review artifact's 9x-tilt claim applies to the paper equations only; tell the user.
- All methods use mu = max_shift = 1.15 (patched in crope/runtime.py). Why: official DyPE and SEGA pipelines do this; the diffusers default gives mu~11.5 at 4096^2 and collapses sigmas.
- Calibration prompts: prompts/calib_pickapic.jsonl (39 filtered Pick-a-Pic prompts), disjoint from the Aesthetic-4K eval set.

## Open questions
- Default 4K shift leaves final evaluated sigma at 0.99974 and one distinct bf16 model timestep; design preserved.
- T2 sign and image-quality benefit require real GPU sampling and remain untested.
- GH200 peak memory and task-size estimates require the smoke timing.
- Crope lacked diffusers, ImageReward, and an Inception backend at last check; no env installs performed here.
