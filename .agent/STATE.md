## Goal
Resume Constancy RoPE from verified evidence; maintain and share an honest working paper.
User first wants confirmation of the project state and restart point before new experiments.

## Status and next action
2026-10-17 CDT takeover: both campaigns COMPLETE; no demonstrated improvement over released SEGA.
Last research commit a1ede1a (October 16, 18:11:57 CDT); earlier uncommitted-state notes are stale.
Live sacct: CR13 / 1076092 completed October 16 at 18:06:13 CDT.
Spool: 0 pending, 0 running, 255 done, 5 older failures, 25 held/archived files; STOP present.
No direct project job in current scheduler records. Generic workspaces were not probed remotely.
Second campaign: only sega_ctm eta 1 passed development; held-out n=187, seed 0, rows 8-194.
IR +0.008575 [-0.008414,+0.027420]; full MUSIQ -0.208599 [-0.708869,+0.315118].
PickScore -0.075259 [-0.102335,-0.049140]; crop MUSIQ -1.131657 [-1.387535,-0.902193].
CLIP -0.093916 [-0.196506,-0.003680]; remaining metric CIs cross zero.
Paper now compiles an empirical working report. Prior proposal sections retained but excluded.
All ten paired means/CIs independently recomputed exactly from archived per-prompt CSVs today.
PDF: paper/constancy_rope_working_2026-10-17.pdf. Build directory on project scratch.
No new GPU work, scorer reruns, installs, SSH, submissions, cancellations, or scratch deletion.
NEXT: confirm takeover and restart point with user; then frame a revised development diagnostic.

## Map
| File | Read it when |
| HANDOVER.md | resuming project ownership and checking operating constraints |
| notes/main2/REPORT.md | reviewing development decision and full held-out results |
| notes/main2/heldout_per_prompt.csv | re-deriving completed campaign statistics |
| notes/main2/analysis_protocol.json | checking frozen reporting boundaries |
| notes/paper_verification_20261017.json | checking current paper table provenance |
| ../paper/update_evidence.py | rebuilding verified current LaTeX table |
| notes/alias_temp.md | reviewing candidate formulas and frozen selection rules |
| notes/results.md | inspecting earlier campaign checkpoints and held tasks |

## Decisions
- Preserve the negative outcome; there is no positive efficacy or SOTA result.
- Development rows select variants only. Do not tune on the inspected held-out set.
- CIs crossing zero mean unresolved differences, not equivalence; nominal per metric.
- New candidate has one held-out seed; no new replication or human study is claimed.
- Keep paper crux on work and bulk images on scratch; delete nothing.
- Retain old theory sources for audit, exclude stale settings from current PDF.
- No SSH, sbatch, scancel, worker edits, or new experiments during takeover review.
- Repository has no remote; preserve changes in a local commit.

## Open questions
- Which revised mechanism merits a new development-only diagnostic after user reviews this checkpoint?
- Earlier theoretical and literature claims need audit before a submission-ready manuscript.
