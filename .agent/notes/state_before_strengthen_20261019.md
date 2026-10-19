## Goal
Develop a genuinely novel training-free DiT resolution method competitive with recent SOTA.
User authorizes unattended project ownership, experiments, paper work and parallel agents.

## Status and next action
2026-10-18 late. Paper v1 DONE (commit fdee66d; paper/dist PDF + zip sent to user). STRENGTHENING PHASE:
- Codex5 (pid 2750090, brief .agent/notes/codex_brief_5.md, log scratch logs/codex5.log): prereg (.agent/notes/prereg_strengthen.md),
  E1 fresh150 confirmation (6 methods, prefix 80_), E2 seeds 1-2 on held-out 8-194 for 4 arms (81_), E3 Qwen-Image (70_/82_),
  eval 90_, then paper update. Codex never submits jobs.
- Fleet CR22 job 1080046 (gh 8 nodes 24h, IDLE_MAX 4h) submitted; holds nodes until queue drained.
- Fable supervisor review of paper running -> .agent/notes/fable_review_2026-10-18.md. User: use Fable as supervisor for
  critique/verification/judgment as needed; use any idle nodes/GPUs found (own allocations) to finish work; keep checking.
NEXT: hourly: check CR22 start, Codex alive, my idle nodes (slurm-idle.sh); when results land, Fable verifies before paper update.

## Map
| File | Purpose |
| HANDOVER.md | Current context and historical campaign boundaries |
| notes/continuation3/dev/selection.json | Completed CR14 four-control results |
| notes/continuation3/independent_review.json | Independent32image/provenance/score audit |
| notes/response_cr17/report.json | Accepted two-prompt response campaign |
| notes/response_cr17_interpretation.json | Independent response interpretation |
| notes/response_cr17_realized.json | Secondary BF16-coordinate analysis |
| notes/response_next_design.json | Stop scalar method branch; no extra transfer sweep |
| notes/observer_shift_cpu.json | Tiny fixed-gain invariance and negative control |
| notes/observer_feedback_cpu.json | Tiny gain-mediated output change |
| notes/next_mechanism_derivation.json | Architecture proof, scope and recommendation |
| notes/rotary_kernel_prior_art_20261017.md | AReA/LayerNorm and phase/covariance collisions |
| notes/post_cr20_mechanism_design.json | Exact provisional phase operator, limits and six-image pilot |
| notes/post_cr20_prior_art.json | PMA/CircleRoPE/Untwisting/TACA collisions; narrow composition only |
| notes/attention_operator_protocol.json | Frozen CR19 collection and analysis |
| notes/attention_cr19/report.json | Accepted192case operator analysis |
| notes/attention_cr19_independent.json | Independent exact replay, hashes and equations |
| notes/attention_cr19/decision.json | Stop conditional-mean branch |
| notes/observer_cr18/report.json | Accepted pretrained synthetic-state contrasts |
| notes/observer_cr18_independent.json | Independent exact saved-tensor recomputation |
| notes/observer_cr18/witness_retention.json | Durable24file tensor archive on work |
| notes/observer_pretrained_report_preflight.json | Validated CPU analysis command and source hashes |
| notes/native_cr20/visual_summary.json | Completed8pair primary/secondary/aux annotations and no-target decision |
| notes/native_cr20_independent.json | Independent model/config/input/archive audit and validator gap |
| notes/native_cr20/paper_receipt.json | Current10page paper with full paired screen |
| notes/post_observer_research_decision.json | Reject underidentified covariance/gain proposal |
| notes/cr14_visual_review.json | Eight fixed-order baseline previews; no causal failure established |
| notes/observer_pretrained_protocol.json | Frozen CR18 causal audit protocol |
| notes/observer_pretrained_submission.json | Job1077211 and watcher registration |
| notes/watcher_cr14.json | Watcher process, registration and lease instructions |
| notes/watcher_ack.json | Actual wake receipts |
| notes/continuation3/confirmation_plan.md | Frozen unused fresh64 |
| notes/main2/REPORT.md | Earlier187prompt negative campaign |

## Decisions
- All prior frozen sources/protocols/results remain unchanged; no consumed pilot reruns.
- CR14 is a completed diagnostic. A gate pass does not justify fresh64 or a novel-method claim.
- Existing octave-11 response is weak; resolved-frequency controls remain untested, not refuted.
- Observer-null sensitivity is a causal hypothesis; centering/energy replacement alone is not novel.
- CR18 adds two no-op repeats to the suggested8forward design, making10 total for numerical validity.
- No amplitude, sigma or seed sweep after seeing the observer outcome.
- First two campaigns remain negative; do not tune on previously inspected held-out rows8-194.
- Preserve scratch. Paper-critical evidence stays on work. W&B offline state is explicit.
- Local-only repository has no remote. Commit meaningful changes; no push destination exists.

## Open questions
- Does the observed synthetic-state sensitivity have any natural-trajectory quality consequence?
- Can a distinct mechanism improve natural generation beyond SEGA and strong recent methods including AReA?
- How can any improvement satisfy novelty and matched-cost efficacy before using fresh64?
