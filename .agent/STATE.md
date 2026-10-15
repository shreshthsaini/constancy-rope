## Goal
Build and CPU-validate the resumable FLUX resolution-extrapolation codebase in design.md.
Research generations and paper claims follow separate GPU validation.

## Status and next action
2026-10-15 02:05. CR1 ran 23:39-01:11 on 4 nodes but all 3 tasks failed at load (IncompleteSnapshotError: root repo files never downloaded); workers idled 90 min and exited. FIXED: runtime.load_pipeline now loads the local snapshot dir. CPU end-to-end check running (logs: /scratch/09032/saini_2/constancy_rope/cpucheck/run.log; calibrate native 1 prompt 2 steps + sample 512^2 2 steps for 6 methods).
Jobs: CR3 1072630 enlarged to gh 4n 24h (pending). Account at 42 queued jobs (submit cap), so no new submissions possible.
NEXT: if cpucheck passes, mv taskq/failed/* -> taskq/pending/ and commit. Then on node start: smoke timing, calib -> pilot -> probe go/no-go.

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
