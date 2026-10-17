## Goal
Develop a genuinely novel training-free DiT resolution-extrapolation method competitive with recent SOTA; maintain and share the evidence and paper.
User clarified full ownership: continue experiments, evaluation, and paper autonomously.

## Status and next action
Unattended CR14 watcher active: tmux crope-watch-cr14, PID1105783, i618-051 / host job1073260.
See notes/watcher_cr14.json and scratch watcher_cr14/state.json for lease, heartbeat, stop, and delivery receipts.
Direct idle-thread wake via local app-server; scheduler120s / files60s; nine tests passed; no model calls while waiting.
Direct WAKE_TEST succeeded end to end; notes/watcher_ack.json records actual operator receipt.
Implementing the same-state response diagnostic from notes/response_probe_plan.json.
Register each next campaign before waiting; watcher persists after CR14; migrate controller before expiry.
2026-10-17 CDT takeover: first two campaigns COMPLETE; no demonstrated improvement over released SEGA.
Last research commit a1ede1a (October 16, 18:11:57 CDT); earlier uncommitted-state notes are stale.
Live sacct: CR13 / 1076092 completed October 16 at 18:06:13 CDT.
Spool: 0 pending, 0 running, 255 done, 5 older failures, 25 held/archived files; STOP present.
CR14 job1076857 is the new project allocation; unrelated allocations were left unchanged.
Second campaign: only sega_ctm eta 1 passed development; held-out n=187, seed 0, rows 8-194.
IR +0.008575 [-0.008414,+0.027420]; full MUSIQ -0.208599 [-0.708869,+0.315118].
PickScore -0.075259 [-0.102335,-0.049140]; crop MUSIQ -1.131657 [-1.387535,-0.902193].
CLIP -0.093916 [-0.196506,-0.003680]; remaining metric CIs cross zero.
Paper now compiles an empirical working report. Prior proposal sections retained but excluded.
All ten paired means/CIs independently recomputed exactly from archived per-prompt CSVs today.
PDF: paper/constancy_rope_working_2026-10-17.pdf. Build directory on project scratch.
New work is isolated in taskq_dev3; old campaign outputs and STOP remain unchanged.
NEXT: monitor CR14 diagnostics; investigate signed SEGA response and a distinct mechanism after the recent-prior-art audit. No automatic confirmation for a diagnostic pass.

Third diagnostic submitted: spatial RMS uniform/normalized controls and lag-aligned ctm, eta1.
Fresh SEGA baseline prespecified; 32 dev images total. Full CPU suite 256 passed in 45.93 seconds, two threads.
CR14 job1076857 is PENDING Priority; no estimated start. Initial gb attempt rejected at qgb submit cap.
Frozen source commit395ba85; spool taskq_dev3 has four ready workflows, old taskq unchanged.
Fresh64 confirmation prompts remain unused; require novelty review and a new fixed method/baseline protocol before use.
Confirmation generator/analyzer prepared and tested; no confirmation tasks or allocation submitted.
The primary adjustment and complete-pair validation were fixed while CR14 remained pending.
After all four task successes, dev3_launch invokes dev3_finalize to validate scores, log W&B,
archive the dev decision, rebuild the working PDF, and commit only its own outputs/state.
If the run fails, inspect taskq_dev3/failed and logs; no auto retry or hidden variant change.
Paper figure and qualified algebra updated; all16 plotted intervals re-derived exactly.
Fixed-order qualitative panel added from rows 8-10, six verified images; display assets archived on work.
Account balance checked: 8212.919/40000 SUs remaining, burn 79.47%.

## Map
See notes/novelty_20261017.md for the clarified target, recent sources, and response-calibration hypothesis.
| File | Read it when |
| notes/continuation3/confirmation_plan.md | planning conditional fresh-prompt confirmation |
| notes/continuation3/theory_audit.md | restoring qualified theory to the manuscript |
| notes/continuation3/PROTOCOL.md | running the frozen third development diagnostic |
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
- Current user authorizes active project operations. Do not alter old campaign artifacts or other projects.
- Repository has no remote; preserve changes in a local commit.
- User authorizes background Codex/GPT-6 Astra/GPT-6.1 Sol agents; higher effort for critical work, within guard limits.

## Open questions
- CR14 remains diagnostic only; no new GPU results yet. Recent AReA and SCG are direct novelty collisions.
- Earlier theoretical and literature claims need audit before a submission-ready manuscript.
