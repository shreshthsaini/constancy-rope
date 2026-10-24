# Strengthening campaign status (analyst view)

Updated 2026-10-24 06:50 CDT by the second analyst (Claude Opus 5.5 subagent). It never submits, cancels or
ssh-es; it publishes task scripts into taskq/pending and analyses on CPU. Stopped at the 6-hour rule: the fleet
never started, so everything is staged.

## State

- Generation complete and validated for every Krea block: factorial 374, tsweep 165, fresh 900, seeds 638
  (+110 reused seed-1 images from out_krea/seed1), ckpt_matrix 32, default_shift 8.
- 22 evaluation tasks (90_eval_*, 90_ir_*) published 2026-10-24 00:45 CDT by
  `scripts/strengthen_evaluation.py <block>_queue.json --references <refs> --publish` (aesthetic4k refs; fresh uses
  prompts/fresh150_refs.jsonl). All `bash -n` clean. Receipts: `evaluation/<block>/status.json`.
- Fleet CR26 (job 1096129, 3 GH200, 12 h) PENDING (Priority, no start estimate) from 00:45 to 06:50 CDT. Pending 22,
  running 0, done 0. Expected cost once running: about 4 s per 4K image, 2,117 images, about 1 h wall on 3 workers.
- Qwen E3 not run (60 failed 82_ tasks preserved; diagnosis in qwen_failure_diagnosis.json).

## FROZEN until every block is merged

Each 90_ task first runs `sha256sum -c evaluation/<block>/dependencies.sha256`, which covers
scripts/strengthen_analysis.py, strengthen_evaluation.py, eval.py, imagereward_score.py, strengthen_diagnostics.py,
crope/utils.py, crope/metrics/*.py and the frozen generation sources. Do not edit them; the merge re-checks too.
Verified unchanged at 06:50 CDT. New code lives in scripts/strengthen_confirm.py (not covered).

## Done

- CPU dry run of the full chain on mock metrics with real identities and sidecars (all blocks complete, counts
  935/275/900/1122; tables compile; figures pass the style audit). Mock files only in the session scratchpad.
- P2 lattice measured from real spectra (see RESULTS.md): dev/Krea 12.0x [7.8, 18.6] at g3.5, 7.4x [4.5, 12.7] at
  g4.5; shared-bin guidance ratio 0.52x on dev, 0.89x on Krea.
- Paper staging: paper/sec/results_confirmation.tex (guarded, macro-driven), paper/tables/strengthen_tables.py
  (ckpt table extended with the shared-bin rows), paper/figs/strengthen/fs_confirm.py (new), paper/INTEGRATION.md.

## Next, in order (successor)

1. When 90_ tasks reach taskq/done: `crope_py scripts/strengthen_confirm.py run` (merges, runs
   strengthen_analysis.py P1/P3/E1/E2 and the diagnostic reports into analysis/<block>, exports CSVs to
   eval_krea/strengthen/, refreshes the auto part of RESULTS.md). Check `strengthen_confirm.py status` first.
2. `crope_py paper/tables/strengthen_tables.py`; then the three fs_*.py figure scripts; look at every PDF.
3. Rewrite the hand-written summary in RESULTS.md from the new CSVs; compare evaluator versions in the new
   metrics.json with eval_krea/main/metrics.json (old seed-0 and seed-1 scores are reused in E1 pooling and E2).
4. Text owner follows paper/INTEGRATION.md. Commit and push (nothing from this session is committed).
