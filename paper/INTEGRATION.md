# Integrating the strengthening campaign into the paper

For the text owner. Written 2026-10-24 by the strengthening analyst. Nothing here edits main.tex or
any section you own; you decide where things go.

## State on 2026-10-24 06:50 CDT

- All Krea generation is complete and validated. The 22 evaluation tasks (prefix 90_) sit in
  taskq/pending. Fleet CR26 (job 1096129) had not started after 6 hours, so no new metric exists yet.
- Measured already (spectra need no evaluator): the checkpoint-by-guidance lattice
  (`.agent/notes/strengthen/analysis/ckpt_matrix_lattice2x2/`). Its table and paragraph still wait
  for the ckpt_matrix evaluation, because the table generator reads the completion-gated report.
- Qwen-Image (E3) did not run: 60 of 60 tasks hit CUDA OOM next to a second 45 GiB process on the
  same GPU (`.agent/notes/strengthen/qwen_failure_diagnosis.json`). Report it as not run.

## Files ready for you

| File | What it is |
| --- | --- |
| `sec/results_confirmation.tex` | Drop-in section `\label{sec:confirm}`. Each paragraph is guarded by its block's `st@done` macro and prints nothing until that block is analysed. With no data it prints the intro and the Qwen paragraph only. Compiles cleanly with and without the macros. |
| `tables/strengthen_tables.py` | Writes `tables/num_strengthen.tex` (macros `\N{st@...}`) and `tables/tab_strengthen_*.tex` (bare booktabs tabulars; the caption, ending in "Key result:", is the `% Caption:` comment at the top of each file). |
| `figs/strengthen/fs_factorial.py`, `fs_sweep.py`, `fs_confirm.py` | Figures, PDFs written beside the scripts. Palette and markers from `figs/PALETTE.md`. Each refuses an incomplete block. |

Table labels the section expects: `tab:strengthen-factorial`, `tab:strengthen-factorial-full`,
`tab:strengthen-sweep`, `tab:strengthen-fresh-tests`, `tab:strengthen-fresh-arms`,
`tab:strengthen-seeds`, `tab:strengthen-ckpt` (plus `tab:strengthen-factorial-means` if you use it).
Figure labels: `fig:fs-factorial`, `fig:fs-confirm` (both inside the section), and `fs_sweep.pdf`
if you want the sweep curve (no float is placed for it yet).

## Steps

1. Preamble, after `\input{tables/num_results}`: `\InputIfFileExists{tables/num_strengthen}{}{}`.
2. `\input{sec/results_confirmation}` after `sec/results_more` (or after `sec/results_decomp`).
3. Place the tables: main text `tab_strengthen_factorial`, `tab_strengthen_fresh_tests`; appendix
   `tab_strengthen_factorial_full`, `tab_strengthen_sweep`, `tab_strengthen_fresh_arms`,
   `tab_strengthen_seeds`, `tab_strengthen_ckpt`. Copy each caption from the file's first comment.
4. Replace the `% STRENGTHEN FACTORIAL` and `% STRENGTHEN SWEEP` placeholders in
   `sec/results_decomp.tex` with a pointer to `\cref{sec:confirm}`, or move those paragraphs there.
5. Re-read every guarded paragraph once its numbers print. The sentences quote signed values and
   copied verdict words; a sentence that reads oddly for the actual outcome must be rewritten from the
   numbers, never the reverse.

## Claims to revisit once results exist

- Any sentence that says base plus temperature "ties" released SEGA: cite the H1 outcome
  (`\N{st@h@H1}`). Only "supported" allows an equivalence statement, and the pooled set reuses the 187
  inspected prompts, so say "pooled extension", not "independent confirmation".
- "The allocation adds full-frame MUSIQ": cite H2 (`\N{st@h@H2}`), fresh prompts only.
- F5 checkpoint caption and text: the guidance confound is now measured. At matched guidance
  FLUX.1-dev carries 12.0x [7.8, 18.6] the Krea lattice power at 3.5 and 7.4x [4.5, 12.7] at 4.5;
  guidance 4.5 scales the dev lattice by 0.52x [0.41, 0.64] and Krea's by 0.89x [0.67, 1.11]
  (8 development prompts, descriptive). These come from `analysis/ckpt_matrix_lattice2x2/summary.csv`
  and the mock-metric dry run of the diagnostic report; quote them through the macros once
  `strengthen_tables.py` has the real report.
- Limitations: add Qwen not run, H1 pooled with inspected prompts, and surrogate metrics only.
- Never write that a method beats SEGA unless a Holm-marked interval in these tables says so, and
  then name the metric.

## Regenerate (project root, OMP_NUM_THREADS=8, crope python)

```
scripts/strengthen_confirm.py status        # per-block 90_ task states
scripts/strengthen_confirm.py run           # merge finished blocks, analyses, CSV exports, RESULTS.md tables
paper/tables/strengthen_tables.py           # macros and tables
paper/figs/strengthen/fs_factorial.py; paper/figs/strengthen/fs_sweep.py; paper/figs/strengthen/fs_confirm.py
```
