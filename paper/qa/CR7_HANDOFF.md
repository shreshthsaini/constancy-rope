# Camera-ready pass, 2026-10-19

## Goal
Resolve the supervisor's camera-ready defects while preserving the findings and all statistical caveats.
Edits are confined to paper/. No commits, queue operations, new experiments or confirmation results.

## Status and next action
Complete. Body: 9 pages; references: pages 10 to 12; total: 25 pages.
Final PDF and standalone compile-source ZIP are refreshed in paper/dist/.
All final pages were rendered at 110 dpi and individually inspected. No remaining layout defect was found.
The fresh ZIP rebuild has identical text and 110 dpi pixels on all 25 pages.
Future confirmation results require another number, layout and package pass after verified insertion.

## Map
| Record | Use |
| :-- | :-- |
| release_check.json | Final hashes, page checks, render paths and fresh rebuild directory |
| number_audit.py / .json / .md | Recompute numerical spot checks from original scratch CSVs |
| table_regeneration.json | Table regeneration produced byte-identical TeX inputs |
| method_figure_audit.json | Figure 2 font and spacing measurements |
| build_cr7_final.log | Final Tectonic build console output |
| package_candidate.json | Standalone package construction receipt |

## Decisions
- Figure 1b has a two-by-two metric layout below the unchanged crops. Each metric has two labelled rows on one axis: base plus temperature minus NTK, and released SEGA minus base plus temperature. The means and paired 95% intervals are checked against table macros. Zero lines, value labels and method colours are direct; there is no stacking, hatching or interval shift.
- Three design rounds were rendered and inspected at 160 dpi with render_pages.sh. The final figure audit is clean. Crop selection is disclosed in one caption sentence.
- Figure 2 stays at full text width. Its minimum text size is 6.4259 PDF pt, including subscripts. The temperature label was spaced farther from the logit-slice label. PDF, editable SVG and exporter agree.
- The six-stage paper/code table and full controller derivations moved to Appendix E. Development diagnostics moved to F, checkpoint figure to A, band-specific details to B, and pilot timing details to G. The forest and spectrum stay as main Figures 3 and 4; the main qualitative grid stays as Figure 5.
- The abstract and repeated discussion were shortened. All distinct original numerical and CI accessors remain in the manuscript or appendices. Nominal intervals, Holm stars and their qualifications remain.
- Table verdicts use neutral ink and direction triangles so method colours keep their figure meaning. Estimate precision agrees across the tables, hero, forest and code-comparison figure. ArXiv identifiers use one bibliography format with no raw arXiv URLs; no existing DOI fields were removed.
- The audit checks 92 scalar values, including 61 directly printed in the abstract, intro and Section 5. No drift was found; raw/archive discrepancies are at most 3.55e-15. Scratch's held-out per-prompt file is absent, so the audit reconstructs it from original main, mainsb and main2 score CSVs and verifies the complete work archive.
- Final TeX logs have no overfull or underfull boxes, unresolved references or missing characters. All internal PDF destinations resolve; no Type 3 fonts were found. Forbidden prose and punctuation scans passed.

## Open questions
No camera-ready blocker remains. Scientific limits remain explicit: the historical result bundles base and temperature, uses one model family and resolution with limited seeds, and has no human study. The checkpoint diagnostic is confounded by guidance. The other process owns pending confirmation results; this pass added none.
The ZIP contains the manuscript and generated compile inputs. Figure regeneration scripts and audit receipts remain in the working paper tree.
