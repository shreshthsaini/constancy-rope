# Constancy RoPE handover

2026-10-15 fix update: band density, empty-band interpolation, squared YaRN gain,
official SEGA defaults, and sega_official_static are implemented. Full validation:
125 tests passed in 22.96 seconds using the crope uv environment and
OMP_NUM_THREADS=8. Test outputs and caches were redirected to /tmp.
Native profiles and traces from the old summed-power implementation require
recalibration. Official static profiles use calibrate.py --mode sega-static
--method sega_official; pilot-static now includes both static SEGA variants.
Probe records include DFT counts for density-preserving octave aggregation.
Changed implementation files: crope/{spectral,gain,rope,runtime}.py and
scripts/{calibrate,enqueue,eval,probe_mtf,sample}.py. Regression coverage is in
tests/{test_gain,test_calibration,test_cli,test_tooling,test_probe,test_runtime}.py;
tests/conftest.py now respects supplied thread settings. The runtime test double
also supports the scheduler config API already called by production.
The working tree stays uncommitted. No queue, Slurm, or paper files were changed.

The earlier implementation notes below predate the scheduler/environment updates
recorded in .agent/STATE.md. Their campaign and dependency status was not rechecked.

Implementation is complete and CPU-validated on the shared node. Read `.agent/STATE.md`,
then `.agent/notes/design.md` for the method specification and
`.agent/notes/implementation.md` for implementation evidence and limitations.

Code: `crope/`, `scripts/`, `tests/`. Outputs and upstream checkouts:
`/scratch/09032/saini_2/constancy_rope`. The fleet worker is unchanged.
No tasks have been enqueued and no GPU work has been launched by this session.

Validation: 107 tests passed in 15.14 seconds using the read-only `mend` uv
environment (torch 2.11.0+cu128, diffusers 0.39.0). All seven CLI help paths pass.
The all-stage dry-run plans 347 scripts and leaves the live queue empty.
Receipts: `/scratch/09032/saini_2/constancy_rope/validation/`.
All 195 cached 4K evaluation prompts and reference paths are exported under
`/scratch/09032/saini_2/constancy_rope/prompts/`.

Before a campaign, inspect `diagnostics/default_schedule.json` on project scratch.
The specified default schedule has final evaluated sigma 0.999739885 at 4096x4096
and only one distinct bf16 model timestep. Upstream baselines change the shift;
this implementation deliberately preserves the design's scheduler setting.
GH200 memory, image quality, real T2 behavior, and learned metrics remain untested.
At the last dependency check, the concurrently built `crope` env still lacked
diffusers, ImageReward, and an Inception backend. HPSv3 is optional and needs
additional cached assets. No environments were modified by this session.

The official SEGA implementation differs from the supplied reference. `sega`
preserves the requested reference algorithm; `sega_official` is a separate
baseline operating on projected image features. See `crope/THIRD_PARTY.md`.

Local commits are blocked by this session's read-only `.git` mount. The attempted
`git add` and `git commit` failed when creating `.git/index.lock`. Source edits
remain in the working tree for review. Existing requirements-lock and dispatch
note edits belong to the concurrent setup session and have been preserved.
