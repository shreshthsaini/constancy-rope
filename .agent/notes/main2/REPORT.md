# Second-campaign result

Completed 2026-10-16T18:08:51.678538-05:00. Neither candidate establishes an improvement over released SEGA under the registered protocol.

Implemented alias_temp, alias_temp_sref, and sega_ctm in the gain controller, runtime hooks, sampler,
and enqueue CLI. Alias gains use the requested axial effective-pool formula; sref matches geometric-mean
logit gain s^0.16. sega_ctm multiplies released SEGA allocation by the time-matched correction and adds
no temperature. Its default eta is 0.5; eta 1.0 is the only variant selected for held-out evaluation.
Derivation and limits: [alias_temp.md](../alias_temp.md). Full CPU suite: 194 passed in 16.61 seconds.

## Dev selection only

Eight prompts, rows 0-7, 4096x4096, seed 0. These numbers select variants and are not held-out results.
Require IR delta > 0 AND full-res MUSIQ delta > 0 AND CLIP delta >= -0.3; at most two variants.

| Variant | ImageReward | Full-res MUSIQ | CLIP | Advance |
|---|---:|---:|---:|---|
| sega_official | 1.423018 | 42.488060 | 30.320004 | Reference |
| alias_temp | 1.281098 | 35.807356 | 29.366986 | No |
| alias_temp_sref | 1.277426 | 44.972845 | 30.238460 | No |
| sega_ctm_eta05 | 1.441334 | 42.480210 | 30.200250 | No |
| sega_ctm_eta10 | 1.448742 | 42.505928 | 30.342451 | Yes |

Only sega_ctm eta 1.0 passes. Eta 0.5 misses MUSIQ by 0.007850; both alias variants lose ImageReward.
No settings changed after selection. Decision: [selection.json](../dev2/selection.json).

## Held-out result

Prompts 8-194, 187 matched seed-0 pairs against released sega_official. Generation included row 7 to
preserve the 27-chunk structure; all analysis excludes it. Krea transformer, SEGA RoPE base, guidance 4.5,
28 steps, and executed noise schedules match the baseline. All 188 sidecars passed the protocol/gain audit.

Deltas are candidate minus SEGA. CIs use 2,000 paired bootstrap draws, seed 0, 2.5/97.5 percentiles.
Higher is better except spectrum errors and tiling, where lower is better.

| Metric | SEGA mean | sega_ctm eta 1 mean | Paired delta [95% CI] |
|---|---:|---:|---:|
| ImageReward | 1.18994 | 1.19852 | +0.00857 [-0.00841, +0.02742] |
| PickScore | 23.05255 | 22.97729 | -0.07526 [-0.10234, -0.04914] |
| CLIP | 29.13552 | 29.04160 | -0.09392 [-0.19651, -0.00368] |
| MUSIQ, full resolution | 45.22784 | 45.01924 | -0.20860 [-0.70887, +0.31512] |
| MUSIQ, 1024 crops | 58.32778 | 57.19612 | -1.13166 [-1.38754, -0.90219] |
| CLIP-IQA, full resolution | 0.60299 | 0.60593 | +0.00294 [-0.00018, +0.00609] |
| CLIP-IQA, 1024 crops | 0.58597 | 0.58846 | +0.00250 [-0.00070, +0.00563] |
| Spectrum log RMSE | 1.06838 | 1.06596 | -0.00242 [-0.00905, +0.00433] |
| Spectrum slope error | 0.19995 | 0.19741 | -0.00254 [-0.00678, +0.00192] |
| Tiling at 1024 | 0.06836 | 0.06844 | +0.00008 [-0.00072, +0.00081] |

ImageReward and full-res MUSIQ have CIs crossing zero. PickScore and crop MUSIQ regress; CLIP also has
a negative nominal CI. No held-out gain is demonstrated. A CI crossing zero does not establish equivalence.
CIs are per metric and are not adjusted for multiple comparisons. This new candidate has one seed;
no seed-1 replication was run. Aggregate FID/KID in the raw 188-image evaluation include overlap row 7
and are not part of this held-out table or the conclusion.

## Evidence and execution

- [Paired table](heldout_paired_vs_sega_official.csv), [means](heldout_means.csv),
  [per-prompt data](heldout_per_prompt.csv), [analysis protocol](analysis_protocol.json).
- [Final raw scores](raw/per_image.csv), [ImageReward](raw/imagereward.csv),
  [score validation](../main2_score_check.json), [generation audit](../main2_generation_check.json).
- All 27 generation chunks and final eval/IR tasks succeeded. STOP was touched at 18:05:35 CDT only
  after the entire pending and running queues were empty. All four workers exited by 18:06:06.
  No jobs were submitted or cancelled here.
- CPU and GPU prepasses reused identical keyed metrics. Verified merge: [receipt](../main2_cache_merge.json).
  CPU W&B logging initially failed because sandbox sockets were blocked; subsequent CPU prepasses used
  local receipts. Final GPU evaluation logged offline W&B run n8xhe6iy. No remote sync is claimed.
- Independent raw-CSV review reproduced all ten means and paired CIs within 1e-10.
  [Final verification](final_verification.json) also records worker exits and the clean diff check.
- Implementation hashes and task manifest: [manifest](../dev2_manifest.json).
  No commit was made because .git is read-only. The original Constancy held-out failure is preserved.

## Files changed by this session

- crope/gain.py, crope/runtime.py.
- scripts/sample.py, scripts/enqueue.py, scripts/heldout_analysis.py; new scripts/dev2_analysis.py.
- tests/test_gain.py, tests/test_cli.py, tests/test_tooling.py; new tests/test_dev2_analysis.py and tests/test_heldout_analysis.py.
- HANDOVER.md, .agent/STATE.md, .agent/notes/results.md; new alias derivation, campaign manifests,
  validation scripts/receipts, and dev2/main2 evidence under .agent/notes.
- Scratch task scripts: four 13_dev_* tasks, eight 43_* dev scorers, 27 53_* held-out chunks,
  and three 62_* scoring/prepass tasks. CPU preprocessing logs are also preserved on scratch.

The untracked slurm/fleet_CR11.sbatch, CR12, and CR13 files were created externally and are not changes
made by this session. Worker and node-launch scripts were not edited.
