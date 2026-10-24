# Strengthening campaign status (analyst view)

Updated 2026-10-24 afternoon CDT by the final analyst (Claude Opus 5.5 subagent). CPU only: no ssh, sbatch,
scancel or broad kills; nothing on scratch was deleted; nothing was committed or pushed.

## State: COMPLETE (Qwen not run)

- All 22 evaluation tasks (90_eval_*, 90_ir_*) are in taskq/done; 0 failed. `scripts/strengthen_confirm.py run`
  merged every block (merge_status.json complete for factorial, tsweep, fresh, seeds, ckpt_matrix, default_shift)
  and wrote complete analyses to analysis/{P1,P3,E1,E2,ckpt_matrix,default_shift}. Exit 0, no script fix needed.
- CSV exports: /scratch/09032/saini_2/constancy_rope/eval_krea/strengthen/analysis_*.csv (20 files).
- Outcomes: H1 supported (pooled 337), H2 supported (fresh 150), H3 withdrawn; fresh-only PickScore equivalence not
  established. Details and every number: RESULTS.md; paste-ready lines: DIGEST.md.
- Verification (verification/): independent recomputation from raw CSVs matches all 223 contrast cells and the
  registered family within 1e-14; 71 of 71 paper macros match the CSVs and 67 of 69 appear verbatim in a scratch
  build of the integrated paper (the other 2 are unprinted macros). Evaluator versions equal the original campaign.
- Qwen E3: not run (60 of 60 82_ tasks OOM next to a co-resident ~45 GiB process; qwen_failure_diagnosis.json).

## Paper files (owned by this analyst; text agent integrates)

- paper/sec/results_confirmation.tex: final, unguarded; inputs the two main-text table floats.
- paper/tables/strengthen_tables.py writes num_strengthen.tex (1,660 macros), tab_strengthen_*.tex (9 tables),
  float_strengthen_*.tex (one float per table) and float_strengthen_appendix.tex (appendix section app:confirm).
- paper/figs/strengthen/fs_factorial.pdf, fs_sweep.pdf, fs_confirm.pdf (audit clean; rendered and inspected).
- paper/INTEGRATION.md: where to \input, superseded sentences, summary paragraph.

## Next (other owners)

1. Text agent: follow paper/INTEGRATION.md (preamble line, two \input lines, placeholder comments, superseded
   sentences); rebuild and check page budget (the section adds about 3.5 pages).
2. Figure agent: hero panel (a) still shows base + temp. as one bar; the factorial splits it (temperature carries it).
3. Commit and push once the user or root session allows it (git push needs SSH to GitHub).
