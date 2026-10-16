## Goal
COMPLETE: implement and test alias_temp, alias_temp_sref, and sega_ctm eta 0.5/1.0 under the fixed protocol.
All four dev runs finished; selected sega_ctm eta 1.0 finished held-out. No improvement over SEGA demonstrated.

## Status and next action
2026-10-16T18:08:51.678538-05:00. Final report: notes/main2/REPORT.md. Full suite: 194 passed in 16.61s.
Dev selection only: both aliases fail IR; eta 0.5 narrowly fails MUSIQ; only eta 1.0 passes all gates.
Held-out: 187 paired seed-0 prompts 8-194, Krea 4K, 28 steps, guidance 4.5, SEGA RoPE.
IR delta +0.008575, 95% CI [-0.008414,+0.027420]; full-res MUSIQ -0.208599 [-0.708869,+0.315118].
PickScore -0.075259 [-0.102335,-0.049140]; crop MUSIQ -1.131657 [-1.387535,-0.902193].
CLIP -0.093916 [-0.196506,-0.003680]. Other metric CIs cross zero; see full ten-metric table.
All 188 generated images passed protocol/gain audit; final scores complete and provenance matches baseline.
All 27 generation tasks, GPU prepass, and final scorers succeeded. No pending/running tasks at completion.
STOP touched 18:05:35 CDT; all four workers exited by 18:06:06. No scancel or new submission.
Small score tables and raw scorer reports archived on work in notes/main2; full images remain on scratch.
CPU prepass W&B socket failure is documented; final GPU evaluation logged offline run n8xhe6iy.
All local CPU sessions reaped. No commit (.git read-only). No additional experiments queued.
Independent review reproduced all ten paired means/CIs within 1e-10; no discrepancies.
Final verification: notes/main2/final_verification.json. git diff --check passed.
NEXT: none for this completed campaign; preserve the negative result and wait for a new explicit task.

## Map
| File | Purpose |
| notes/main2/REPORT.md | complete dev table, held-out table, conclusion, changed files |
| notes/dev2/selection.json | exact frozen dev decision and eight IDs |
| notes/main2/heldout_paired_vs_sega_official.csv | paired 95% CI table, including sega_ctm_v2 |
| notes/main2/heldout_per_prompt.csv | archived per-prompt inputs, dev rows excluded |
| notes/main2_generation_check.json | all 188 source/profile/seed/schedule/gain checks |
| notes/main2_score_check.json | complete scores, provenance, offline W&B, STOP time |
| notes/alias_temp.md | formula derivation, limits, B3 composition, frozen protocol |
| notes/dev2_manifest.json | source hashes and task paths |
| notes/results.md | prior evidence and all campaign checkpoints |

## Decisions
- Both new candidates fail to demonstrate a held-out improvement; original Constancy negative remains valid.
- Dev values are selection only. No retuning on held-out and no paper/SOTA claim.
- Only eta 1.0 advanced; eta 0.5 and aliases have no new held-out runs.
- CIs use 2,000 paired bootstrap draws, seed 0; nominal per metric, not multiple-comparison adjusted.
- CI crossing zero means unresolved difference, not equivalence. New held-out candidate uses one seed only.
- Preserve code changes in the tree, small evidence on work, and bulk outputs on scratch; delete nothing.
- No SSH, sbatch, scancel, installs, worker/node-launch edits, or broad kills.
- OMP/BLAS threads 8; own processes under 20. External CR11/12/13 fleet files are not our changes.

## Open questions
- None required for this campaign. A new research direction requires a separate explicit task.
