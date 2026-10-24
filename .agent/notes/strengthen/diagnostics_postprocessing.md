# P2/P7 post-generation reporting prepared 2026-10-19 UTC

No new diagnostic images or scores were available when these sources were
prepared. No report tables or replacement figures were generated. No tasks were
queued. Frozen generation, evaluator, worker and launch sources were unchanged.

## Sources and completion gate

- `scripts/strengthen_diagnostic_report.py`: complete-block descriptive reports.
- `paper/figs/f5_checkpoint_matched.py`: replacement matched-guidance figure.
- `tests/test_strengthen_diagnostic_report.py`: lightweight CPU contracts.

The reporter reads the frozen `ckpt_matrix_queue.json` or
`default_shift_queue.json` and requires all expected PNG/JSON pairs and all
generation task completion records with matching task hashes. P2 requires all
32 images: rows 0 through 7, seed 42, both checkpoints and guidance 3.5/4.5.
P7 requires all eight `sega_default_shift` images: the same rows, seed 0.
It invokes the evaluation gate's existing sidecar/configuration/schedule
validators, checks all frozen generation source hashes, verifies full image
dimensions and PNG SHA256, and preserves sidecar and PNG hashes in the report.
Duplicate, missing or substituted identities are rejected.

P7 also validates all eight existing `sega_official` Krea pilot images and their
sidecars against the archived generation task. This comparator uses guidance
4.5, 28 steps, prompt-derived seed 0 and mu 1.15. Its transformer revision is
verified through the pinned snapshot in the archived task. Source revision and
hardware can differ from P7. Actual SDPA dispatch was not recorded, so the old
attention label does not establish backend identity. Timing is not a causal
performance comparison.

Scores are joined by exact PNG path, with method, prompt, resolution and seed
cross-checks. Required metrics are ImageReward, PickScore, CLIP, full-frame and
1024-crop MUSIQ, full-frame and 1024-crop CLIP-IQA, spectral log-RMSE, spectral
slope error and tiling1024. Every selected value must be finite. CSV hashes,
prompt and generation-manifest hashes, image witnesses and reporter source hash
are retained. The existing pilot CSVs can contain other methods; only the eight
validated SEGA paths enter the comparison.

## Fixed descriptive summaries

Report every arm mean and every fixed paired contrast over all eight development
prompts. Intervals use 2,000 resamples, NumPy generator seed 0, paired prompt as
the sampling unit and nominal 95% percentile intervals with linear quantiles.
P2 contrasts are dev minus Krea at each guidance, guidance 4.5 minus 3.5 within
each checkpoint, and the difference between those two guidance effects. P7 is
default shift minus official shift. No hypothesis verdict, multiplicity-adjusted
significance mark, tuning rule or general failure rate is computed. These
development diagnostics are outside the preregistered confirmatory family.

P2 additionally preserves the original F5 numerical definition. Use native
RGB/255 luminance weights (0.2126, 0.7152, 0.0722), subtract the image mean,
multiply a separable Hann window, compute abs(FFT)^2 / n^2, average radial power
using 49 geometric edges from 2/n to 0.5 and multiply by squared geometric bin
centers. Select the dev maximum in the strict band 1/64 < f < 1/8. Divide dev by
Krea power in that same bin. At each guidance report the geometric mean of the
eight ratios and the percentile bootstrap interval on their logarithms. This is
the original `peak_ratio` definition; the broader annotation band formerly used
inside the old panel is not reused. The peak is selected using dev, which can
favor dev in the ratio. It is a descriptive spectral contrast, not an unbiased
estimate of a lattice effect. All 32 spectra and all 16 paired ratios are saved.

P7 also records each image's last evaluated sigma and the number of evaluated
sigmas below 0.3, directly from the validated sidecar arrays.

## Commands after generation and both scorers finish

Use the existing crope uv environment. These commands do not schedule work.
Each output directory must be new; existing reports are preserved and rejected.

```bash
export OMP_NUM_THREADS=8
/work/09032/saini_2/vista/library/uv-envs/crope/bin/python scripts/strengthen_diagnostic_report.py ckpt_matrix --out /scratch/09032/saini_2/constancy_rope/analysis/strengthen/ckpt_matrix_v1
/work/09032/saini_2/vista/library/uv-envs/crope/bin/python scripts/strengthen_diagnostic_report.py default_shift --out /scratch/09032/saini_2/constancy_rope/analysis/strengthen/default_shift_v1
/work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/figs/f5_checkpoint_matched.py --report /scratch/09032/saini_2/constancy_rope/analysis/strengthen/ckpt_matrix_v1/report.json
```

Default score paths are
`scratch/eval_krea/strengthen/<block>/{per_image.csv,ir.csv}`.
The optional `--eval-root` changes only that location. Outputs are `report.json`,
`per_prompt.csv`, `means.csv` and `paired.csv`, plus P2 `lattice_summary.csv` and
`lattice_per_prompt.csv`. No output directory is created until all evidence and
statistics pass validation. An absent block exits 2 with a missing-count message.

## Matched-guidance figure rule

The figure requires the complete P2 report, validates all 32 image/sidecar hashes,
all input hashes, the current reporter hash and the exact frozen development
prompt order. It checks all 32 spectral arrays and re-derives the eight-prompt
spectral summaries before displaying them. It renders only row 0, fixed before
generation: two rows for guidance 3.5 and 4.5, dev and Krea columns, then spectra
for that displayed pair on shared axes. It never searches or selects prompts.

Every image uses the original F5 fixed native 192-pixel crops:
A=(1952,928,2144,1120), B=(1952,2976,2144,3168), centered at
(2048,1024) and (2048,3072). Full frames use Lanczos reduction to 768 pixels;
crops are unchanged native RGB pixels and PDF/SVG encoding stays lossless.
The eight-prompt spectral summaries appear only in caption/table metadata, not
inside an individual image panel. Dynamic summary wording reports whichever
direction is observed. Outputs are `paper/figs/f5_checkpoint_matched.pdf`, `.svg`,
`.json` and `.caption.txt`. Metadata preserves crop boxes and raw RGB hashes,
displayed image witnesses, spectral limits, all prompt ratios and summaries.
Existing F5 source and artifacts remain available. Integrate and render-inspect
the replacement only after actual results exist.

## Validation completed before data

Nineteen CPU tests passed for complete fixed designs, missing/duplicate/substituted
identities, exact score joins, nonfinite values, paired constant-difference
intervals, geometric summaries, spectral parity against independent masked radial
means, strict peak-band endpoints, evidence hash changes, spectral completeness,
caption-summary recomputation and fixed prompt order. Tests generated no figures.
Both new sources also passed Python compilation.

Live readiness checks rejected P2 with 32 missing image pairs and eight unfinished
tasks, and P7 with eight missing image pairs and two unfinished tasks. Neither
check created its proposed report directory. A separate read-only check validated
all eight historical P7 comparator images and joined all ten fixed metrics for
all eight. The replacement figure cannot yet be rendered or visually inspected.
