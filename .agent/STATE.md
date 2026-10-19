## Goal
Strengthen finished findings paper fdee66d with E1 fresh150, E2 seeds1/2 and E3 Qwen.
Report all outcomes, revise claims, rebuild/inspect PDF and refresh standalone source zip.

## Status and next action
Updated 2026-10-19T04:06:24+00:00.
Preregistration frozen; E1 metadata/exclusions in progress. Generation queue code ready.
165 existing seed1 outputs validated; no new generation queued yet.
CR22 job1080046 last observed PENDING at04:00:51Z; next poll>=04:10:51Z.
NEXT: Finalize fresh150 caption audit and refs, publish E1 then E2; begin Qwen only afterwards.
Live machine-readable progress: .agent/notes/strengthen/live.json.

## Map
| File | Read when |
| .agent/notes/prereg_strengthen.md | Exact frozen hypotheses, selection, statistics and amendments |
| .agent/notes/strengthen/live.json | Current queue counts, receipts, next steps and scheduler observation |
| .agent/notes/strengthen/*_queue.json | Frozen task identities, scripts and source hashes |
| .agent/notes/strengthen/seed1_reuse.json | Reused existing165 held-out seed1 images |
| .agent/notes/strengthen/fresh_selection* | Dataset selection, exclusions and reference provenance |
| .agent/notes/state_before_strengthen_20261019.md | Historical state before this task |
| paper/PLAN.md | Paper honesty, formatting and evidence requirements |
| .agent/notes/main2/REPORT.md | Existing negative results and raw evidence |
| .agent/notes/results.md | Append-only campaign progress |
| slurm/fleet_CR22.sbatch | Existing8-GH200/24h fleet; no submissions authorized here |

## Decisions
- H1 joint IR/Pick equivalence; H2 full-MUSIQ superiority; H3 IR improvement ceiling0.03.
- E2 averages seeds0/1/2 within prompt before bootstrap; n187, not561; seeds1/2 sensitivity.
- No selection/tuning on new outputs. Incomplete blocks cannot establish hypotheses.
- Preserve prior seed1 and all historical results; verify before reuse.
- Shared CPU host: OMP8 and own processes<20. No local GPU work.
- Never ssh,sbatch,scancel,pkill,killall; no scratch deletion; worker scripts unchanged.
- .git read-only: leave edits, no commits. No em dashes or section symbol in prose.
- CR22 pending: scheduler polls at least10minutes apart; sleep when no useful work.

## Open questions
- GPU completion and final hypothesis outcomes remain pending unless recorded above.
- Qwen transfer needs official settings, tiny CPU tests and a successful GPU smoke first.
- Metrics are surrogate evidence; no human-preference, universal-transfer or SOTA claims.
