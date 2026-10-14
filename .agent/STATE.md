## Goal
Build and CPU-validate the resumable FLUX resolution-extrapolation codebase in design.md.
Research generations and paper claims follow separate GPU validation.

## Status and next action
Implementation complete: upstream ports, gains, sampling, calibration, probe, metrics, and task tooling.
Final CPU suite: 107 passed in 15.14 seconds under mend (torch 2.11.0, diffusers 0.39.0).
All CLI helps pass; 195 cached eval prompts exported; all-stage dry-run plans 347 scripts with an empty live queue.
No enqueueing, scheduler operations, SSH, or GPU runs by this implementation session.
Local commits blocked by the read-only .git mount; source remains reviewable in the working tree.
External setup previously reported fleet 1071114 pending and env/download work; not rechecked here.
NEXT: review the default-schedule diagnostic, finish the external crope env build, then run GPU smoke separately.

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

## Open questions
- Default 4K shift leaves final evaluated sigma at 0.99974 and one distinct bf16 model timestep; design preserved.
- T2 sign and image-quality benefit require real GPU sampling and remain untested.
- GH200 peak memory and task-size estimates require the smoke timing.
- Crope lacked diffusers, ImageReward, and an Inception backend at last check; no env installs performed here.
