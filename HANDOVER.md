# Constancy RoPE handover

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
