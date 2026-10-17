## Goal
Develop a genuinely novel training-free DiT resolution-extrapolation method competitive with recent SOTA.
User authorizes full unattended project ownership, experiments, evaluation, paper, and background agents.

## Status and next action
2026-10-17 08:04 CDT: CR14 job1076857 RUNNING since07:21:46 on c609-092,c609-151.
CR14baseline/uniform workflowsDONE; normalized/lagRUNNING, bothGPUs100percent, nofailures.
Partial8pairaudit verifies16PNGs/provenance/scores: uniform-minusfresh IR-.01739,MUSIQ-1.56672,
CLIP+.58680,Pick-.08321,cropMUSIQ-.51775. Descriptive only; no selection beforeallfourdone.
CR15 job1077082 FAILED after2m12s: both2048smoke endpoints bitwise equal, same-state replay exact, histories differ.
No4K started. Failure evidence archived in notes/response_cr15_failure; original acceptance rule unchanged.
CR16 job1077122 COMPLETED/0:0 in2m14s. BaselineA/B/D histories drift; all BF16gains/state/velocity/endpoints match.
Prompt1B/C FP32gain maxdifference1.1920928955078125e-07; default identical-feature allocation repeats drift.
Eight isolated deterministic repeats match for both prompts. Evidence archived in notes/response_cr16_repeatability.
CR17 job1077143 COMPLETED at 08:00:25 CDT, elapsed 13m49s on c638-141/142.
Source1a06786; separate frozen notes/response_deterministic_protocol.json; strict original equality checks retained.
Policy determinism applies only inside OfficialSEGA.update; enabled/warn-only flags restored in finally.
All numericalpolicy/source provenance recorded; both gain histories saved. No proven originalbaseline qualityequivalence.
Existing watcher hot registry includes taskq_response_deterministic plus prior queues; never duplicate watcher.
New wrapper+response tests13passed/10s; adapted report helper33tests/15.55s; independent review found no blocker.
Earlier full CPU suite306passed. CR15/CR16 protocol/source hashes unchanged; fresh64unused.
BothCR17 2048smokes PASSED exactno-op/history/endpointchecks; receipt notes/response_cr17_smoke.json.
Both 4K stages also PASSED; full provenance/endpoint/history archive verified in notes/response_cr17.
Axis-octave prediction error median .937 (range .709-1.354); scalar .426 (.050-3.039).
The octave map has limited predictive value; one early scalar case fails. No controller promotion.
Secondary BF16-coordinate analysis complete: axial errors .688-1.339, scalar .029-3.004.
Half-gain off-linearity remains substantial; no quantization-dead intervention. New helper: 12 tests passed.
Next-experiment design under review; no new GPU job.
Use scripts/response_report.py <scratch/response_probe_deterministic/job1077143> --out <freshworkdir>
with --protocol .agent/notes/response_deterministic_protocol.json --submission .agent/notes/response_deterministic_submission.json.
The helper rejects incomplete/failed campaigns; neither numerical repair nor diagnostic pass establishes novelty/SOTA.
No automatic confirmation or promotion from diagnostics to novelty/SOTA.
Existing watcher v3: tmux crope-watch-cr14, PID1581602, i618-051/controllerjob1073260.
Scheduler120s/files60s;33watcher tests. Direct idle-thread wake verified by actual operator receipt.
Hot registry covers subsequent jobs; never duplicate watcher. Register next jobs before returning to waiting.
Controller lease ends12:45:04CDT; watcher alerts one hour before expiry. Arrange verified compute handoff then.
Account balance before CR15 submission7946/40000SUs,~80.14percent burn. Refresh before each new submission.
Paper: paper/constancy_rope_working_2026-10-17.pdf. Active empirical report, no positive efficacy claim.
Prior-art notes: response_prior_art_20261017.md and response_actuator_prior_art_20261017.md.
AFM already edits attentionFourierbands and discussesnonmonotonicity; inversion/reuse alone notnovel.
Potential distinction is attention-actuator-to-Fourier-power response; efficacy and originality remain unproved.
Fresh64 remains unused; current runs use development rows only.

## Map
| File | Purpose |
| HANDOVER.md | Long project context and operating constraints |
| notes/response_deterministic_submission.json | CR17 job, budget and watcher registration |
| notes/response_deterministic_protocol.json | Frozen numerical-policy response probe |
| notes/response_cr16_repeatability/receipt.json | Numerical drift evidence and source artifacts |
| notes/response_repeatability_protocol.json | Separate bounded numerical diagnosis, immutable |
| notes/response_cr15_failure/receipt.json | Failed smoke status and archived evidence hashes |
| notes/response_probe_protocol.json | Frozen two-prompt smoke/4K measurement protocol |
| notes/response_probe_audit.json | Critical review fixes and CPU validation |
| notes/response_prior_art_20261017.md | Full-method/code comparisons and next research tests |
| notes/watcher_cr14.json | Watcher PID, socket, registry, lease and stop instructions |
| notes/watcher_ack.json | Actual autonomous wake acknowledgment |
| notes/continuation3/PROTOCOL.md | Frozen CR14 four-control diagnostic |
| notes/continuation3/partial_baseline_uniform.json | Verified partial8pair audit, no selection |
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
- No controller lease extension or handoff is verified yet; current lease ends 12:45 CDT.

## Open questions
- Do same-state interventions predict smaller unfit perturbations above numerical/quantization noise?
- Can a cheap reusable response model improve final-image quality over SEGA and direct spectral correction?
- How does any eventual mechanism differ substantively from recent AReA,SCG and Jacobian-guidance methods?
- Novelty and matched-cost efficacy must be established before spending the fresh64 confirmation set.
