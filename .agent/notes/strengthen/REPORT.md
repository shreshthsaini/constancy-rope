# Evidence-strengthening progress

This is a campaign handoff, not a completed-results report. The new study is still awaiting GPU execution.

## Execution

CR22 job 1080046: PENDING for Priority at 2026-10-19 04:46:53 UTC; no scheduler start estimate.
Next scheduler poll: no earlier than 04:56:53 UTC. Use `scripts/strengthen_status.py --scheduler`.
All 315 published tasks are intact. These cover 2,117 new FLUX/diagnostic images and three Qwen smoke images.
Full Qwen generation (66 tasks, 450 images) remains gated by a successful matching GPU smoke.

| Block | Published tasks | Pending | Running | Done | Failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| factorial | 54 | 54 | 0 | 0 | 0 |
| ckpt_matrix | 8 | 8 | 0 | 0 | 0 |
| tsweep | 24 | 24 | 0 | 0 | 0 |
| fresh | 132 | 132 | 0 | 0 | 0 |
| seeds | 94 | 94 | 0 | 0 | 0 |
| qwen_smoke | 1 | 1 | 0 | 0 | 0 |
| default_shift | 2 | 2 | 0 | 0 | 0 |

The sweep reuses the two fixed endpoint controls. Seed 1 reuses 110 verified images.
Every generation source checksum still matches its published manifest. Do not edit those sources in place.
No evaluation tasks have been published. Large blocks use method-level evaluator tasks and a verified CPU merge. The combined analysis, evaluation and diagnostic suite passes all 82 tests.

## Registered outcomes

| Test | Frozen decision | Current outcome |
| --- | --- | --- |
| H1 ImageReward | Pooled 337 prompts; 95% CI within +/-0.0819771364 and Holm-adjusted equivalence p <=0.05 | Pending |
| H1 PickScore | Pooled 337 prompts; 95% CI within +/-0.10 and Holm-adjusted equivalence p <=0.05 | Pending |
| H2 MUSIQ | Fresh 150 only; SEGA-minus-base-temperature lower 95% CI >0 and adjusted one-sided p <=0.05 | Pending |
| H3 controller comparison | Removed from new generation; retain completed historical negative results | No new confirmation claim |

The primary Holm family contains the first three elementary tests. H1's pooled extension includes already
observed historical rows and is not wholly independent confirmation. Report fresh-only estimates separately.
All amendments preceded generation and preserve the original preregistration bytes.

## Verified findings available now

- Retrospective Holm audit: 28 of 39 nominally significant comparisons survive in the full 60-comparison family;
  16 of 21 survive in the condensed 30-comparison family. Historical nominal intervals reproduce within 5e-16.
  The allocation MUSIQ effect survives both families; weak CLIP and CLIP-IQA marks do not all survive.
- MUSIQ uses pyiqa 0.1.16 KonIQ10k: 16,384 native 32-pixel patches plus 49 and 144 resized-view patches,
  totaling 16,577 at 4K, with no native patch cap. This is not a direct sharpness measurement.
- Default flow shift mu 11.55 gives 28 BF16 model timesteps all equal to 1; official mu 1.15 retains 28 distinct
  model timesteps. This is a numerical schedule audit; image-quality evidence is pending.
- Qwen CPU parity, temperature and cache tests passed; pinned model files are downloaded and size-verified.
  No successful GPU smoke or transfer result exists yet.

## Manuscript and reproducibility

Known-evidence prose and figures are revised. Table3 is in the appendix; both historical tables have Holm marks.
The main text includes signed absolute paired gains, the spectrum panel and a documented two-row crop grid.
Mechanism claims and the guidance confound are qualified. MUSIQ preprocessing and hero selection are explicit.
F2's illustrative formula no longer double-counts temperature. F7 shows three individual prompt matrices.

Current preview: `/scratch/09032/saini_2/constancy_rope/paper_preview/strengthen_review/main.pdf`, 27 pages,
clean Tectonic build with no overfull boxes or undefined references. Page review and source-package compilation
have been performed; pagination will be finalized after inserting new results. `paper/dist` is unchanged.
`paper/package_source.py` packages only active dependencies and creates a deterministic, hash-verified ZIP.
Offline W&B campaign registration: zndwlu7k; no remote sync is claimed.

## Resume

1. Read STATE, live.json and queue_status.json, then respect the next scheduler-poll time.
2. Inspect actual smoke/failure receipts and task logs. Publish full Qwen only after a matching pass.
3. Use ANALYSIS_AND_EVALUATION.md to publish evaluation for each complete generation block and merge shards.
4. Run the five registered analysis manifests. Use diagnostics_postprocessing.md for P2/P7 reports and figure.
5. Insert all complete outcomes, including failures or unresolved tests, into the paper. Rebuild, inspect every
   rendered page, then preserve old dist artifacts and refresh the PDF and standalone ZIP.

No SSH, submissions, cancellations, broad kills, scratch deletion, worker changes or commits are authorized.
All sessions must be reaped before ending. Keep OMP_NUM_THREADS=8 and own processes below 20.
