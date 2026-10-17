## Goal
Develop a genuinely novel training-free DiT resolution-extrapolation method competitive with recent SOTA.
User authorizes full unattended project ownership, experiments, evaluation, paper, and background agents.

## Status and next action
2026-10-17 ~07:33CDT: CR14 job1076857 RUNNING since07:21:46 on c609-092,c609-151.
Both GPUs100percent; no failed workflows; two workflows active, two pending.
CR15 response diagnostic job1077082 submitted, PENDING Priority;2GH200nodes/45min,2dev prompts.
Spool taskq_response and exact output/receipt paths registered in watcher_cr14/registrations.json.
Response source frozen at6d14fe5; notes/response_probe_protocol.json immutable for this run.
Audit fixed gain-vector RMS wording,2048smoke coverage, fixed operating-point replay, provenance and timing.
CPU suite306passed/47.73s; independent critical recheck found no blocker; GPU validation remains unrun.
Full baseline/probed endpoints and histories must match before any response analysis is accepted.
Next: verify CR14 results when complete; evaluate CR15 no-op, prediction accuracy, response sign/coupling.
No automatic confirmation or promotion from diagnostics to novelty/SOTA.
Existing watcher v3: tmux crope-watch-cr14, PID1581602, i618-051/controllerjob1073260.
Scheduler120s/files60s;33watcher tests. Direct idle-thread wake verified by actual operator receipt.
Hot registry covers subsequent jobs; never duplicate watcher. Register next jobs before returning to waiting.
Controller lease ends12:45:04CDT; watcher alerts one hour before expiry. Arrange verified compute handoff then.
Account balance before CR15 submission7946/40000SUs,~80.14percent burn. Refresh before each new submission.
Paper: paper/constancy_rope_working_2026-10-17.pdf. Active empirical report, no positive efficacy claim.
Recent prior-art comparison: notes/response_prior_art_20261017.md; generic Jacobian/control ideas established.
Potential distinction is attention-actuator-to-Fourier-power response; efficacy and originality remain unproved.
Fresh64 remains unused; current runs use development rows only.

## Map
| File | Purpose |
| HANDOVER.md | Long project context and operating constraints |
| notes/response_submission.json | CR15 job, budget, spool hashes, watcher registration |
| notes/response_probe_protocol.json | Frozen two-prompt smoke/4K measurement protocol |
| notes/response_probe_audit.json | Critical review fixes and CPU validation |
| notes/response_prior_art_20261017.md | Full-method/code comparisons and next research tests |
| notes/watcher_cr14.json | Watcher PID, socket, registry, lease and stop instructions |
| notes/watcher_ack.json | Actual autonomous wake acknowledgment |
| notes/continuation3/PROTOCOL.md | Frozen CR14 four-control diagnostic |
| notes/continuation3/dev/completion.json | Expected validated CR14 finalizer receipt |
| notes/continuation3/confirmation_plan.md | Conditional unused fresh64 protocol |
| notes/main2/REPORT.md | Completed negative held-out campaign |
| notes/main2/heldout_per_prompt.csv | Archived187paired prompt-level measurements |
| notes/novelty_20261017.md | Goal and expanded recent-baseline boundaries |

## Decisions
- Preserve frozen CR14 sources/spool and immutable CR15 protocol;20CR14repository hashes verified unchanged.
- CR14 covers fresh SEGA plus spatial RMS uniform/normalized and lag-aligned constancy;32dev images.
- Last successful CR14 worker validates/archives, updates paper/WandB offline and commits via dev3_finalize.
- Live-state wording may differ from finalizer literal replacements; reconcile state after its receipt.
- First two campaigns complete and negative; do not tune on previously inspected held-out rows8-194.
- Second campaign187pairs/seed0:IR+.008575 CI[-.008414,.027420],MUSIQ-.208599[-.708869,.315118].
- PickScore-.075259[-.102335,-.049140],cropMUSIQ-1.131657[-1.387535,-.902193].
- CIs crossing zero do not establish equivalence. No near-SOTA/SOTA claim currently supported.
- Old taskq drained:255done,5olderfailures,25held;STOP remains. Do not revive consumed pilots.
- Preserve scratch; archive paper crux on work. W&B offline evidence is retained.
- Local-only repository has no remote; commit meaningful changes locally.
- No controller lease extension or handoff is verified yet; current lease has over five hours left.

## Open questions
- Do same-state interventions predict smaller unfit perturbations above numerical/quantization noise?
- Can a cheap reusable response model improve final-image quality over SEGA and direct spectral correction?
- How does any eventual mechanism differ substantively from recent AReA,SCG and Jacobian-guidance methods?
- Novelty and matched-cost efficacy must be established before spending the fresh64 confirmation set.
