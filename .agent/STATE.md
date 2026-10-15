## Goal
Build and CPU-validate the resumable FLUX resolution-extrapolation codebase in design.md.
Research generations and paper claims follow separate GPU validation.

## Status and next action
2026-10-15 14:40. KREA campaign live (FLUX.1-dev lattices at 4K even under official DyPE; Krea clean). Nodes: CR3 4n (until 10-16 11:40) + CR9 gh-dev 1n.
Done (out_krea/pilot): ntk2, dype, sega_official, sega, constancy (8 prompts, 4096^2, ~290 s/img; dype 330). Images coherent; constancy comparable to sega_official/dype visually. Krea calib: calib/krea_profile.npz. Probe p0 done (probe_krea/4096x4096).
Running/queued: pilots temp_only, yarn; probes p1 p2; 31 pilot-static (sega, sega_official); 112 main tasks 50_cXXXXX (ntk2, dype, sega_official, constancy x 195 prompts, interleaved by chunk, ~63 GPU-h).
NEXT: when pilots + static done -> enqueue eval pilot (rename to 40_ so it runs before main) and read metrics. Read probe_krea mtf.csv (3 prompts). Perf: our pipeline 290 s/img vs official DyPE 192 s; investigate if time allows.

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
