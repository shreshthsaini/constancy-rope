# Camera-ready numerical audit

Outcome: PASS. 92 scalar numbers across 24 held-out effects, four gain shares, and the seed-1 replication were checked. No paper table generator was imported or run.

The raw scratch CSVs reproduce the retained per-prompt archive to 3.55e-15 and the checked registered paired summaries to 4.44e-16 absolute error. Every checked macro agrees at its printed precision. The checks also cover printed share-table entries and relevant main comparison cells.

eval_krea/heldout/heldout_per_prompt.csv is absent, and its paired summary has 50 comparisons, predating SEGA + TM. Values were reconstructed from original main, mainsb and main2 per_image.csv files plus ImageReward CSVs, then compared with the retained work per-prompt archive, its complete 60-comparison summary, and the scratch summary where present.

Reproduce with `OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/qa/number_audit.py`. Input SHA256 hashes, source locations, unrounded results and individual comparisons are in `number_audit.json`.

| Quantity | Recomputed value | Macro |
| :-- | :-- | :-- |
| ho@d@ntk@ir | -0.594 | tables/num_results.tex:104 |
| ho@ci@ntk@ir | [-0.701, -0.494] | tables/num_results.tex:106 |
| ho@d@ntk@musiq | -15.74 | tables/num_results.tex:119 |
| ho@ci@ntk@musiq | [-17.50, -14.00] | tables/num_results.tex:121 |
| ho@d@ntk@musiqc | -18.70 | tables/num_results.tex:124 |
| ho@ci@ntk@musiqc | [-20.68, -16.86] | tables/num_results.tex:126 |
| ho@d@tsb@ir | +0.026 | tables/num_results.tex:204 |
| ho@ci@tsb@ir | [-0.016, +0.070] | tables/num_results.tex:206 |
| ho@d@tsb@musiq | -2.09 | tables/num_results.tex:219 |
| ho@ci@tsb@musiq | [-3.25, -0.96] | tables/num_results.tex:221 |
| ho@d@alloc@musiq | +2.09 | tables/num_results.tex:502 |
| ho@ci@alloc@musiq | [+0.96, +3.25] | tables/num_results.tex:504 |
| ho@d@alloc@iqa | +0.009 | tables/num_results.tex:536 |
| ho@ci@alloc@iqa | [+0.001, +0.018] | tables/num_results.tex:538 |
| ho@d@dype@ir | -0.164 | tables/num_results.tex:54 |
| ho@ci@dype@ir | [-0.258, -0.087] | tables/num_results.tex:56 |
| ho@d@dype@pick | -0.49 | tables/num_results.tex:59 |
| ho@ci@dype@pick | [-0.63, -0.36] | tables/num_results.tex:61 |
| ho@d@dype@musiqc | +1.97 | tables/num_results.tex:74 |
| ho@ci@dype@musiqc | [+1.16, +2.80] | tables/num_results.tex:76 |
| ho@d@dype@iqa | +0.029 | tables/num_results.tex:79 |
| ho@ci@dype@iqa | [+0.020, +0.038] | tables/num_results.tex:81 |
| ho@d@sctm@ir | +0.009 | tables/num_results.tex:254 |
| ho@ci@sctm@ir | [-0.008, +0.027] | tables/num_results.tex:256 |
| ho@d@sctm@musiq | -0.21 | tables/num_results.tex:269 |
| ho@ci@sctm@musiq | [-0.71, +0.32] | tables/num_results.tex:271 |
| ho@d@sctm@pick | -0.08 | tables/num_results.tex:259 |
| ho@ci@sctm@pick | [-0.10, -0.05] | tables/num_results.tex:261 |
| ho@d@sctm@musiqc | -1.13 | tables/num_results.tex:274 |
| ho@ci@sctm@musiqc | [-1.39, -0.90] | tables/num_results.tex:276 |
| ho@d@sctm@clip | -0.09 | tables/num_results.tex:264 |
| ho@ci@sctm@clip | [-0.20, -0.00] | tables/num_results.tex:266 |
| ho@d@ctm@clip | -0.27 | tables/num_results.tex:164 |
| ho@ci@ctm@clip | [-0.49, -0.04] | tables/num_results.tex:166 |
| ho@d@ctm@musiq | -3.04 | tables/num_results.tex:169 |
| ho@ci@ctm@musiq | [-4.28, -1.81] | tables/num_results.tex:171 |
| ho@d@ctm@musiqc | -2.19 | tables/num_results.tex:174 |
| ho@ci@ctm@musiqc | [-2.81, -1.57] | tables/num_results.tex:176 |
| ho@d@ctm@tile | -0.0068 | tables/num_results.tex:199 |
| ho@ci@ctm@tile | [-0.0107, -0.0026] | tables/num_results.tex:201 |
| ho@d@ctmtsb@musiqc | -2.08 | tables/num_results.tex:662 |
| ho@ci@ctmtsb@musiqc | [-2.43, -1.75] | tables/num_results.tex:664 |
| ho@d@ctmtsb@musiq | -0.95 | tables/num_results.tex:657 |
| ho@ci@ctmtsb@musiq | [-1.67, -0.22] | tables/num_results.tex:659 |
| ho@d@ctmtsb@spec | -0.015 | tables/num_results.tex:677 |
| ho@ci@ctmtsb@spec | [-0.021, -0.008] | tables/num_results.tex:679 |
| ho@d@ctmtsb@slope | -0.006 | tables/num_results.tex:682 |
| ho@ci@ctmtsb@slope | [-0.010, -0.002] | tables/num_results.tex:684 |
| ho@a@alloc@musiq | 2.09 | tables/num_results.tex:503 |
| sh@p@ir | 104 | tables/num_results.tex:444 |
| sh@ci@ir | [97, 113] | tables/num_results.tex:445 |
| sh@p@pick | 103 | tables/num_results.tex:461 |
| sh@ci@pick | [97, 109] | tables/num_results.tex:462 |
| sh@p@clip | 101 | tables/num_results.tex:478 |
| sh@ci@clip | [90, 112] | tables/num_results.tex:479 |
| sh@p@musiqc | 99 | tables/num_results.tex:512 |
| sh@ci@musiqc | [96, 103] | tables/num_results.tex:513 |
| ho@nprompts | 187 | tables/num_results.tex:1605 |
| so@n@tsb | 55 | tables/num_results.tex:985 |
| so@d@tsb@musiq | -2.79 | tables/num_results.tex:922 |
| so@ci@tsb@musiq | [-4.58, -0.95] | tables/num_results.tex:924 |
| so@agree | 17 | tables/num_results.tex:1092 |
| so@cells | 20 | tables/num_results.tex:1093 |

Intervals here retain the declared bootstrap seed and prompt order. Borderline interval verdicts are not equivalence tests or proof that another bootstrap draw would give the same sign.
