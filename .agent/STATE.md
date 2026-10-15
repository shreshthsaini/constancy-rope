## Goal
Build and CPU-validate the resumable FLUX resolution-extrapolation codebase in design.md.
Research generations and paper claims follow separate GPU validation.

## Status and next action
2026-10-15 11:55. CR3 1072630 RUNNING since 11:40 on c610-072,c619-102,c637-021,c637-022 (gh, 24h -> ends 10-16 11:40). Other CR jobs still pending (CR4,5,7,8,9).
Done: 00 smoke (1024^2 9.2 s; 4096^2 291 s/img, peak 40.8 GB, sigmas OK mu=1.15), 10 calib (calib/flux_dev_profile.npz, 16 prompts).
Running/pending: 20 probe, 30 pilots (ntk2, dype, constancy, yarn, sega_official, sega, temp_only), 8 prompts each at 4096^2 (~40 min/task).
NEXT: 4K outputs show lattice artifacts (see notes/results.md). Waiting on 05_diag official DyPE+dev; Krea downloading. Probe extras held in taskq/hold.

## Map
| note | read it when |
| notes/design.md | implementing or changing any method, baseline, or metric |
| notes/implementation.md | reviewing validation, deviations, and operational limitations |
| ../crope/THIRD_PARTY.md | inspecting upstream revisions, licensing, and SEGA differences |
| notes/codex.md | reviewing the original implementation dispatch |

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
