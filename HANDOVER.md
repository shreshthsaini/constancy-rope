# Constancy RoPE: handover (read this first)

Last takeover update: 2026-10-17 CDT. Live short state: `.agent/STATE.md`. Result log: `.agent/notes/results.md`.
Design spec: `.agent/notes/design.md`. Codex briefs: `.agent/notes/codex_brief_{1,2,3}.md`. Review of the SEGA paper that
started this: `../research_reviews/sega_2605.22668_2026-10-14.md` (+ artifact https://claude.ai/artifact/Phv2pGDvZDZwpmNM3Qh8GL).

## Evidence strengthening (current, 2026-10-19 UTC)

Last update: 2026-10-20T01:51:38+00:00. Active task confirms the finished findings paper.
Read `.agent/STATE.md`, immutable `.agent/notes/prereg_strengthen.md` and
`.agent/notes/strengthen/live.json` for current state and resume details.
CR22 PENDING at 2026-10-20T01:51:38+00:00. 315 pending, 0 running, 0 done, 0 failed, 0 missing. Preparation and the 82-test integration checks are complete. See queue_status.json and live.json for the current observation; final results and paper release depend on complete verified outputs.
Next: Next scheduler query no earlier than 2026-10-20T02:01:38+00:00. Inspect new smoke/output/failure receipts. Release full Qwen only after a valid pass; evaluate complete blocks, merge shards and run the registered analyses. Sleep while the fleet is pending.
Current session forbids SSH, submissions/cancellations, worker edits, scratch deletion and commits.
All older entries below are historical and do not override the current task.

## Gold rubric self-score (text agent)

2026-10-19, build `paper/main.pdf` (not `paper/dist/`): 27 pages; Sections 1 to 8 and the Reproducibility
statement on pages 1 to 10, references from page 11, appendix contents page 14. Clean build: no overfull boxes,
no undefined references. Intro 721 words, 33 sentences, median 21, mean 21.8, max 32 (none over 35). Abstract
200 words. 22 of 22 spot-checked macro values (shares, deltas, CIs, compute) match the compiled main text.
Figure captions follow figs/CAPTIONS.md (figv3); table shading follows figs/PALETTE.md (only the SEGA (released)
row or an ablation default, #E6F0F6). Scores: 0 below, 1 meets, 2 above. Figure-content items rely on the figure agent.

| Item | Score | Note |
| :-- | :-: | :-- |
| Abstract 130-200 words, headline number, URL | 0 | 200 words and numbers; no public URL yet (release sentence instead) |
| Hero under abstract, full width, panel-walking caption | 1 | |
| Footnotes (at most two, administrative) | 1 | none |
| Question by end of para 1; motivating float beside para 2 | 1 | Table 1 wraptable |
| "We introduce" para, numbers in para 4, three "We" bullets | 1 | last bullet carries 104% [97, 113] |
| Median sentence under 22, none over 35 | 1 | median 21, max 32 |
| Method diagram with lettered panels and appendix zoom | 1 | Fig. 2 and Fig. E.1 |
| Equations introduced by a clause, "where" line | 1 | |
| Boxed side-by-side algorithms with Input/Output, appendix pointers | 1 | Algorithms 1-2, top of page 4 |
| Setup run-ins with counts, steps, compute hours | 2 | adds a main-text cost table (Table 6) |
| Main table: booktabs, arrows, shading, bold best, Key result | 2 | adds paired-verdict triangles and Holm stars |
| Finding paragraphs with bold claims and numbers | 1 | quotes paired deltas with CIs rather than before/after pairs |
| Ablation tables side by side, default shaded, appendix versions | 1 | Tables 3-4 |
| Plots: palette, markers, value labels, shared legends, >= 6 pt | 1 | figure agent; not re-measured at print size here |
| Qualitative grid, same prompt and seed, labeled rows | 1 | Fig. 5 |
| Confounds named; limitations with concrete scope | 2 | eight-item main-text Limitations section |
| Reproducibility statement mapping details to appendices | 2 | plus script-generated numbers and checklist (App. H) |
| Appendix: contents, zoom, banded config table, expansions, prompts | 1 | samples are the first rows, labeled as such, not random |
| Floats within one page, >= 3 wraps, <= 60% float pages | 1 | four wraps; page 6 is near the 60% limit |
| Captions above tables, below figures, bold leads | 1 | |
| Same colors across hero, plots, tables | 1 | |
| No em dashes, no section symbol, no figure text under 6 pt | 1 | |

Mean 25/22 = 1.14, below the 1.2 ship bar, with one 0 (no public URL until release). Below the bar or open:
the URL; main-table finding paragraphs quote deltas rather than before/after values; page 6 float share
(Table 2 plus the Figure 4 wrap) sits near 60%; Figure 4 prints before Figure 3 because the wrap is in-text.

## Paper status (2026-10-18, camera-ready pass)

- Honest findings paper, `paper/main.tex` -> `paper/main.pdf` (22 pages: body Sections 1-7 = 9 pages, references from page 10,
  appendix A-H). Build: `cd paper && ~/.local/bin/tectonic -X compile main.tex`. Framing rules: `paper/PLAN.md`.
- Main-text figures show held-out results only; partial Constancy (n=41) and dev-only Alias-T numbers live in Appendix D.
  Appendix floats are numbered per section (Figure A.1, Table B.1, ...).
- Numbers: every result number is a macro from `paper/tables/make_tables.py` (and `compute_table.py`, `mechanism_numbers.py`);
  an independent audit recomputed 600+ macros from the raw CSVs (scripts in `/scratch/09032/saini_2/tmp/finalcheck/audit/`).
  Borderline verdicts: SEGA + TM CLIP (upper CI -0.0037) and seed-1 crop CLIP-IQA for SEGA base + temp. depend on bootstrap seed 0.
- Deliverables (gitignored `paper/dist/`): `constancy_rope_paper.pdf` and `constancy_rope_paper_src.zip` (standalone, tested
  with tectonic in a fresh directory; not tested with pdfLaTeX).

## Current takeover checkpoint (2026-10-17 10:44 Pacific)

- At10:44Pacific,5minexpiry alert: replacementCPU1077574 andGPU1077335 still PENDING; no destinationREADY/release/agentreceipt. Actual submittedCPUbatchSHA and all controller/CR21pins match. Durablewatcherstate/registry snapshot and recovery instructions: `.agent/notes/controller_migration_blocked.json`. Source ends10:45:04Pacific; queued GPUworkers are self-contained, but monitoring/chat may pause until automaticCPUfallback starts. This is BLOCKED migration, not a verified handoff.
- The user authorizes full unattended project ownership, experiments, evaluation, paper work, and parallel agents. The target is a genuinely novel method competitive with recent SOTA. Neither novelty nor competitive efficacy is established yet.
- CR14 job **1076857 completed** at 08:18:51 CDT in57m05s. All four workflows succeeded. Finalizer archived scores, logged W&B, rebuilt the paper, and committed c056ce9. Independent review verified all32 images, raw scores, sidecars and26 frozen entries; recomputation differs by at most2.5e-16. Receipt: `.agent/notes/continuation3/independent_review.json`.
- Only RMS-normalized allocation passes the frozen development gate: IR+0.036285, MUSIQ+0.388861, CLIP-0.065491, but PickScore-0.066567 and cropMUSIQ-0.214963. Uniform and lag-aligned controls fail. This modest control result does not overturn previous held-out negatives or establish a novel method. Fresh64 is unused. Preserve the completed CR14 source/spool/protocol; do not rerun it.
- The completed baseline/uniform pair was independently audited: all16 PNGs, identities, source hashes, schedules and scores match. Uniform-minusfreshSEGA on8 development prompts: IR -0.017389625, MUSIQ -1.56671834, CLIP +0.58680439, PickScore -0.08320761, crop MUSIQ -0.51774937. These are descriptive partial results. Receipt: `.agent/notes/continuation3/partial_baseline_uniform.json`.
- CR17 job **1077143 completed** at 08:00:25 CDT in13m49s. All four smoke/4K stages passed independently rechecked provenance, exact no-op replay, full gain-history equality, and endpoint equality. Accepted archive: `.agent/notes/response_cr17/`; independent interpretation: `.agent/notes/response_cr17_interpretation.json`. The delayed prompt0 watcher event has been acknowledged; do not rerun this completed campaign.
- CR17 measured two development prompts. Axis-octave half-step prediction error has median0.937, range0.709-1.354, with9/12 below the zero-predictor error1. Scalar median0.426, range0.050-3.039, with5/6 below1. Selected octave-11 lies below the token-grid DFT floor; this does not test all rotary frequencies. Unequal actuator norms prevent direct response-size comparisons. Each4K trajectory added39 transformer calls. Numerical success is not efficacy or novelty.
- Secondary BF16-coordinate analysis, `.agent/notes/response_cr17_realized.json`, only modestly changes errors. Axial half-vector departures from the fitted control direction are31-49 percent, scalar11-16 percent; none of72 interventions vanished through quantization. The new helper passes12 tests and agrees with independent review. Do not invert the current octave map.
- Decision: stop scalar-as-new-method work. A fixed temperature pulse offers no concrete novelty advance, so the proposed12 extra trajectories are not justified. `.agent/notes/response_next_design.json` records this decision and its exploratory calculations. CR14 is now complete; resolved-frequency controls remain untested, not refuted.
- CPU observer audit: the standard FLUX prenorm path ignores common-channel residual shifts in real arithmetic, but SEGA reads that channel mean. Seven fixed-gain tests and seven separate feedback tests pass on miniature random FLUX. In the feedback audit, changed gains differ by0.11748 and output by0.00717; fixed gains and gain-mediation control agree within1.34e-15. Evidence: `observer_shift_cpu.json`, `observer_feedback_cpu.json`, and `next_mechanism_derivation.json`. No pretrained/BF16/quality claim.
- Prior-art boundary: AReA already rejects channel-mean cancellation and uses a shift-invariant channel-std sensor; LayerNorm centering is established. See `rotary_kernel_prior_art_20261017.md`. The narrow new causal question is SEGA feedback converting a fixed-gain null perturbation into an output change; an invariant statistic alone is not a new method.
- CR18 **job1077211 COMPLETED** at08:51:16CDT, elapsed2m36s, two GH200 nodes c636-061/c636-111. Both prompts completed all5 forwards. Official CPU report and independent saved-tensor review accept exact byte repeats, input/provenance, gains, observer/target and contrast checks. Archive `.agent/notes/observer_cr18/`; independent review `observer_cr18_independent.json`. Source061fa03 and frozen protocol b98de615 remain unchanged.
- CR18 gain-only output L2 differences109.986/66.181 exceed fixed-gain shift residuals16.505/16.448 by6.664/4.024. Adaptive/gain-only cosines.98781/.96289. Mediation residuals remain15.62/27.07percent of gain-only; interaction21.48/36.64percent. Both have106 changed BF16 gain coordinates. This supports gain-mediated observer sensitivity at two artificial sigma.5 endpoint/noise states, not exact invariance, natural-trajectory harm, quality or novelty. Do not sweep or repeat this consumed diagnostic.
- CR18 reportSHA127ef1c2a2b5b27e590a1d80ee7baead59bf6ec9375c59a4f5616dcbef84df49. Preserved24 tensor witnesses including CR17 endpoint dependencies,139505284bytes, at `/work/09032/saini_2/vista/results/constancy_rope/observer_cr18_job1077211`; hash/path map in `observer_cr18/witness_retention.json`. Full conditioning/projected features were not saved and remain runtime receipts. W&B runs are offline. No GPUs remain allocated to this project campaign.
- CR19 **job1077286 COMPLETED** at09:27:35CDT, elapsed2m03s, twoGH200nodes c610-042/c610-051. Fourforwards; both actual-QKV captures pass byte replay and frozen provenance. Source28092cc/protocolSHAa0fb07d23ad61224f275829741287d4a242e862418002f00f8e052bec1081dea unchanged. Official report `.agent/notes/attention_cr19/report.json` accepts all192cases; independent own-equation recomputation agrees within3.91e-14. ReportSHA6ad008611008013eb3066d917d41c55db81cd4b612c61ffda5fa704d63b95fff.
- CR19 descriptive medianrelativeerrors: sparse.313271, zeroorder.039987, globalH.020316, complementaryH.019229, weightedH.011648. Complementary improves138/worsens54; only33absolute-error changes exceed separate BF16gap (26improve,7worsen). Call18 retains large errors. This is a small mixed change in a known PISA expression, not an efficacy/novelty result. **Stop fixed-window conditional-mean branch**, no GPUkernel or quality sweep; decision in `attention_cr19/decision.json`.
- CR19 preserves10QKV/output witnesses,442654958bytes, at `/work/09032/saini_2/vista/results/constancy_rope/attention_cr19_job1077286`; hash map `attention_cr19/witness_retention.json`. Input ancestry remains in CR18durablearchive. W&B7ppsy698/v41n6z5j offline.30s telemetry missed the14sforwardwindow, so no100percentutilization claim; finiteworkersexitednormally. No projectGPUjob remains active.
- CR20 **job1077304 COMPLETED** at09:42:02CDT in2m28s, two GH200 nodes. All8 native1024 renders pass frozen source/protocol, model, sigma, unit-gain and PNG checks. Independent audit also reconstructs every saved initial BF16 input from the declared CPU seed. All40 native/reference/input files (202873493bytes) are preserved at `/work/09032/saini_2/vista/results/constancy_rope/native_cr20_job1077304`. Source4a12688/protocolSHAba80b1ddd2ca794a84e2607be6773acf5253f80402ece5ddf71d53b400775a18 unchanged. RuntimeHEAD06f0c87 only adds protocol/preflight notes. W&B5x7fh624/drwwbu60 offline. No project GPU remains active.
- CR20 independent model reviewers (GPT6Astra and GPT6.1Sol) viewed all16 randomized label-hidden common1024 images before details. Both agree: primary native->4K has7satisfied/satisfied and1violated/unjudgeable (celestial layout); all other transition cells zero. Secondarykneeling improves violated->satisfied; auxiliaryKodak violatedboth,Paradis satisfied->violated. Five originals perreviewer checked; four crops archived; detail changes none, disagreements none. Full annotations/summary in `native_cr20/`. No human study, causal resolution effect or failure-rate estimate. Close this particular semantic-loss rationale; no seedsearch/OCRpivot/fresh64 follows.
- CR20 parent reporter initially compared full sidecarconfig to itself; independent audit verified actual8 fullconfigs and exposed this coverage gap. The NEW UNFROZEN reporter now derives expectations independently;27 corruption tests pass and all8 actual configs pass read-only supplemental validation. `native_cr20/config_supplement.json` preserves old reporter source/hash and unchanged original report/archive. Independent audit remains separate.
- NEXT: preserve controller continuity, then execute/evaluate the frozen provisional image-to-text phase intervention. Design `post_cr20_mechanism_design.json` and primary-source review `post_cr20_prior_art.json` are complete. Endpoint-normalized coordinates map each image axis to0..63 using nativebasefrequencies for cross-modal attention only; fixedmass arm preserves all image-key weights and total text mass at the same input in real arithmetic. Ingredients collide with PMA, Circle-RoPE, Untwisting and TACA; no identical full operator found in the bounded review, but novelty and quality remain unestablished. A scalar-temperature tie-breaking counterexample is only an actuator distinction.
- CR21 job1077335 submitted and registered (protocol7a5d072b frozen, source0eb2161): existingdev0/1,seed0,28steps,guidance4.5,4096square,3arms baseline/fixedmass/unnormalized. All57blocks/all28steps/all512textpositions; no sweep. Native1024 four-step deployed/privatekernel/identity smoke and tiny GPUkernel checks precede4K. LSEbackend allows fixedmass correction with1.538percent extraattention arithmetic (notruntime); unnormalizedcontrol uses an explicit additionalimageonlypass for stablemixture, so has highercost. Isolated processor/worker/scoring implementation passes99CPUtests in both parent and independent runs; all reviewedsourcehashes match. Review `post_cr20_implementation_review.json` has no blocker, actualCUDA remains untested. Querychunk65536 fits declaredmemoryheadroom and avoids16xsmallbranchcalls; premodelGPU smoke includesproductiontextbranchgeometry. Originals remain frozen. SameGPU task will score all3outputs with existingCLIP/Pick/MUSIQ/CLIPIQA andImageReward,bootstrap0,thenexit. Two nodes/40minutes maximum submitted; fresh64 untouched.
- CR21job1077335 is PENDING priority at10:32CDT; existing watcher registration and scheduler observation verified. A separate post-completion reporter `scripts/crossmodal_phase_report.py` passes37tests in both agent and parent runs. Invoke only after both `workflow_scored.json` files exist, using exact command in `crossmodal_phase_report_interface.json`; it validates both tasks before creating a durable work archive and fixed label-hidden previews/crops. Then obtain two independent model readings before revealing mapping/metrics, report all7metrics and measured costs, update paper. If a GPU gate fails, retain failure and original frozen sources/protocol; any corrected replacement needs a new declared protocol and watcher registration.
- Spatial complement prototype independently passes30 tests; optional within-cell-H prefix is distinct from raw-cloud covariance. Report `spatial_complement_cpu_with_cell_h.json` preserves every case and FP32 cancellation. Fullrank H+prefix cost is857MB/24headsFP32; grouped support reuse changes logical four-corner reads from412GB to1.61GB. These are cost estimates, not a GPU benchmark. Fixedrank8 can discard an entire correction. Original raw report/source remain archived.
- Offline capture reporter independently passes13 tests; it checks artifact provenance and byte replay, then reports all192 fixed query cases and all5 controls with a separate SDPA numerical gap. Source `scripts/attention_operator_report.py`. CR19 is complete and independently accepted.
- Equal-cell PISA comparison is complete and independently passes16 tests. Conditional versus global mean errors are.003734/.380833 in the selected-large-H case; heterogeneous attention weights still leave conditional error.099727 versus weighted-cell.001317. All8 declared cases are retained in `pisa_complement_cpu.json`. These are synthetic implementation checks, not quality or novelty.
- Actual-SDPA capture collector passes30 CPU tests, including miniature BF16 FLUX joint/single paths and exact replay. Durable CR18 input preflight passes both prompts and all16 frozen sources. Receipt `attention_operator_capture_preflight.json`. Independent cost and integration reviews found no blocker; CR19 is now complete.
- Attention-measure CPU prototype is complete:15 tests pass; exact union/oracle error<=8.89e-16. A fixed single-cloud covariance case gives relative errors.006950 for complement covariance and.150759 for a global shortcut, but this is PISA-inspired, not a literal PISA reproduction. Bimodal counterexample overshoots dense.999664 with5.333333; approximate partition invariance fails. Evidence `attention_measure_cpu.json`; source `scripts/attention_measure_cpu.py`. No efficacy/novelty claim.
- Prior-art reviews `attention_measure_research.json` and `attention_measure_independent.json` identify Scatterbrain, MomentKV, PISA and Sol-Attn collisions. Conditional spatial statistics remain a possible engineering refinement, not a new family. Equal-cell PISA, spatial-cell cost and capture/report CPU tests are complete; independent reviews support the separately frozen CR19 operator collection. The original prospective design is historical; `attention_operator_protocol.json` now controls execution. The single-cloud and within-cell covariance objects must not be conflated.
- CR15 job1077082 remains FAILED under its original protocol: both2048 endpoints matched, no-op replay was exact, but histories differed; no4K stage started. Archive: `.agent/notes/response_cr15_failure/`. Do not reclassify it.
- CR16 job1077122 completed a separate repeatability diagnosis. Ordinary baselines also drifted in history, while observed BF16 gains, state/velocity hashes and endpoints matched. The largest FP32 gain difference was1.19e-7. Eight isolated deterministic allocation repeats matched for each prompt. This localizes the issue to feature-to-gain arithmetic, not a unique kernel. Archive: `.agent/notes/response_cr16_repeatability/`.
- CR17's separate frozen source1a06786 and `response_deterministic_protocol.json` declare deterministic algorithms only inside OfficialSEGA.update, with flags restored afterward. No general quality equivalence with the original numerical policy is established. CR14 and original CR15 sources remain unchanged.
- One watcher runs in tmux `crope-watch-cr14`, PID2142630, on i618-051/controllerjob1073260. Runtime: `/scratch/09032/saini_2/constancy_rope/watcher_cr14`. Hot registry includes CR15/16/17/18/19/20/21; CR21job1077335 hot-read and PENDING scheduler observation verified; CR14 is also watched. Direct autonomous wake was verified; receipt `.agent/notes/watcher_ack.json`. Never duplicate the watcher. Register new jobs before waiting.
- Controller lease ends **10:45:04 Pacific** (12:45:04CDT). The one-hour alert was handled: no timely GG allocation available; 1h/8h tests forecastOctober19 and generic1076903 forecastsOctober18. New autonomousCPU successor1077574 queuedmaximum8h/minimum1h; handoff-role manifest in watcher runtime, actual submitted script hash verified. Superseded1077465 cancelled whilepending after fixing live-discovered FQDN Slurmfilter bug. Handoff is NOT yet verified, and there may be an interruption after expiry. Watcher now alerts60/15/5min; singleton restartPID2142630 verified after oldPIDexit. Bootstrap/watcher107tests pass; sources pinned by controller_bootstrap_protocol.json and submitted wrapper. Do not edit those pinned sources while queued without updating the declared fallback. Full details controller_bootstrap_submission.json/interface.json. Bootstrap verifies actualallocation/thread/cwd, waits sourceidle release or independentlyverified terminalallocation+stalewatcher+freelock, preservesoutbox, and requests actual resumedhostreceipt. Original chatUI attachment remains unverified. Latest budget7908/40000SU at11:57CDT; refresh before each submission and stop requesting nodes at96 percent burn.
- Current paper: `paper/constancy_rope_working_2026-10-17.pdf`, eleven-page empirical report. Retains all8 CR20 primary/secondary pairs and negative target finding; adds prospective CR21 phase equations, knowningredientcollisions and cost/numerical limits. Clean build, newpages9/10 visually checked. Receipt `crossmodal_phase_paper_receipt.json`. Prior negative heldout/CR14/CR18/CR19 evidence retained; no SOTA claim.
- Post-observer research review rejects covariance-aware gain fitting without an independently justified correction target. DoPE is an additional verified rotary-band Gram-spectrum prior-art collision. Decision: `.agent/notes/post_observer_research_decision.json`. No new GPU sweep or fresh64 use. Resolution-focused review is complete in `resolution_pivot_20261017.json`: no distinct intervention yet; a prompt-grounded native-success/high-resolution-failure target lacks native comparators. No specialized text-repair pivot or new GPU sweep follows.
- All eight CR14 baseline development images were visually inspected in fixed order at768px; no obvious global duplication was seen at that scale. Row5 requested shop text is garbled, but no resolution-specific cause is established. Receipt: `.agent/notes/cr14_visual_review.json`. Do not turn this exploratory observation into a quality or causal claim.
- CPU CR18 report helper passed28 reporter plus23 frozen-worker tests and independently accepted the real campaign. Code review and preflight receipts are retained. Do not overwrite its accepted archive or rerun GPU work merely to repeat validation.
- Prior-art notes: `response_prior_art_20261017.md`, `response_actuator_prior_art_20261017.md`, and `novelty_20261017.md`. AFM already edits attention Fourier bands; finite differences, Jacobian guidance, normalization, inversion and reuse are established. A local response map alone does not establish a new generation method.
- The first two campaigns are complete and negative; their inspected187 held-out prompts cannot become fresh confirmation. Old taskq remains stopped with255 done,5 historical failures and25 held. Do not revive it. Fresh64 remains unused and needs a distinct method plus a recent-baseline protocol before confirmation.
- Repository is local-only with no configured remote. Latest evidence commit:b7cb304 (CR20); CR21source0eb2161, protocol0f88f2d, registration09ecf51. Keep implementation and paper-critical crux on work; bulk artifacts remain on scratch.

## 1. Goal and the user's rules
- Training-free resolution extrapolation for FLUX at 4096^2. Beat released SEGA (arXiv 2605.22668, github rajabi2001/sega)
  with a principled, vision-science-derived per-band RoPE attention gain ("Constancy RoPE").
- User: write a paper ONLY if results are legit and the method beats the baselines (close to SOTA is fine if it beats them).
  Never report dev-set wins as results. Keep this file detailed so a fresh agent can continue after a crash.
- User authorized holding GPUs on any partition and using Codex in the background.

## 2. Verdict so far (do not relitigate without new evidence)
Second campaign COMPLETE (2026-10-16): neither new candidate demonstrates an improvement over released SEGA.
Full report and tables: `.agent/notes/main2/REPORT.md`. Full CPU suite: 194 passed.
Dev only (n=8): both alias variants lose IR; sega_ctm eta 0.5 narrowly loses MUSIQ; eta 1.0 alone passes.
Held-out sega_ctm eta 1.0 (187 paired prompts 8-194, seed 0):
- IR +0.008575, 95% CI [-0.008414,+0.027420]; full-res MUSIQ -0.208599 [-0.708869,+0.315118].
- PickScore -0.075259 [-0.102335,-0.049140]; crop MUSIQ -1.131657 [-1.387535,-0.902193].
- CLIP -0.093916 [-0.196506,-0.003680]. Remaining metric CIs cross zero.
Thus the B3 refinement does not repair the earlier failure. No equivalence claim follows from CIs crossing zero.
All generation/scoring succeeded; exact paired tables and per-prompt data are archived on work.
STOP was touched after pending/running were empty; all four workers exited by 18:06:06 CDT.
Independent review reproduced all ten paired means/CIs within 1e-10. The third diagnostic queued on October 17 is separate from this completed campaign.
The older evidence follows.
Held-out prompts 8-194 (n=187, seed 0) + seed-1 replication on 8-62 (n=55); FLUX.1-Krea-dev, guidance 4.5, 28 steps,
mu = max_shift 1.15; bootstrap 95% CIs, paired vs sega_official (`eval_krea/heldout/heldout_paired_vs_sega_official.csv`).
- constancy_tm on SEGA's RoPE base + SEGA temperature: ties ImageReward/PickScore; WORSE on MUSIQ (-3.0), MUSIQ crop (-2.2),
  CLIP (-0.27), CLIP-IQA, spectral error. Seed 1 same pattern.
- No per-band rule (temp_only on SEGA base): ties IR/Pick/CLIP/MUSIQ-crop; MUSIQ -2.1 worse. So SEGA's released allocation
  gives a small real full-res sharpness gain; most of SEGA's win is RoPE base + global temperature.
- dype: IR -0.16, Pick -0.49 worse, MUSIQ-crop +2.0 better. ntk2 far worse. Paper-equation SEGA is much worse than released;
  its static tilt breaks images (CLIP 13).
- Constancy (ntk2 base, tau_N = ln N_tgt / ln N_train) == temp_only; per-band correction only acted at sigma <= 0.35
  (late-profile target); time-matched target (constancy_tm) fixed activity but not quality.
=> Method hypothesis refuted as designed. Current paper reports this negative evidence;
older theory/proposal section files remain excluded pending review.

## 3. Hard-won facts
- FLUX.1-dev at 4096^2 gives a dot/halftone lattice even under OFFICIAL DyPE code -> use FLUX.1-Krea-dev (both DyPE and SEGA do).
  Loader: env `CROPE_TRANSFORMER=<Krea snapshot>` (Krea transformer + FLUX.1-dev text encoders/VAE). Task scripts export it.
- diffusers default mu extrapolates to ~11.5 at 4K and collapses sigmas; official repos pass max_seq_len = image_seq_len
  (mu = 1.15). Patched in `crope/runtime.py` (register_to_config before configs are recorded).
- Released SEGA code differs from its paper (alloc 1 - 0.15*spread*tanh(1.5 z), clipped >= 1, x_embedder features,
  NTK base / (1 + 0.1 ln s)). `sega_official` = released; `sega` = paper equations. See `crope/THIRD_PARTY.md`.
- Band power must be a grid-independent density (orthonormal FFT, mean per band). Fixed.
- SDPA: prefer cuDNN on GH200 (215 s/img at 4096^2 vs 290 s with forced flash). Peak 41 GB. GPU compute-bound at 100%.
- Blackwell gb nodes (e.g. c672-016) have 4 GPUs; node_launch.sh starts 4 workers; 4096^2 Krea ~97 s/img (2.2x GH200). torch 2.11 cu128 works there. qgb allows 3 submits per user.
- Krea native 1024^2: 5.5 s/img. Calibration (16 Pick-a-Pic prompts, disjoint from eval) -> `calib/krea_profile.npz`,
  time-resolved `calib/krea_profile_tm.npz`.
- ImageReward needs the `mend` env (crope's transformers too new): `scripts/imagereward_score.py`.
- `CROPE_ROPE=<rope method>` env runs any controller on another method's RoPE base (recorded in config["rope"]).
- T2 probe on Krea: raising a band's gain raises that band's power on the width axis (elasticity 0.1 to 1.7), weak on height.

## 4. Layout
Code (git, local only, no remote): `/work/09032/saini_2/vista/Projects/Exploration/constancy_rope`
- `crope/` runtime.py (pipeline, FluxExperiment, resume), gain.py (controllers + attention processor), rope.py (pos-embed
  variants), spectral.py, sega_official.py, metrics/. `scripts/` sample.py, calibrate.py, probe_mtf.py, eval.py, enqueue.py,
  imagereward_score.py, heldout_analysis.py, dev2_analysis.py. `tests/` (194 pass: `OMP_NUM_THREADS=8 <crope python> -m pytest -q tests`).
- `slurm/` fleet_CR*.sbatch, node_launch.sh (per-GPU workers; kills telemetry after workers exit), worker_loop.sh
  (atomic claim, requeue on SIGTERM, exits after 90 min idle or when taskq/STOP exists).
Env: `/work/09032/saini_2/vista/library/uv-envs/crope/bin/python` (torch 2.11 cu128, diffusers 0.39). ImageReward: `.../uv-envs/mend/bin/python`.
Scratch: `/scratch/09032/saini_2/constancy_rope/`
- `taskq/{pending,running,done,failed,hold}` + `STOP` marker (present after the completed second campaign).
- `out_krea/{pilot,main,mainsb,seed1,devtm_eta*,devsb,sweep}/METHOD/4096x4096/*.png+json`; `eval_krea/{pilot,main,mainsb,
  seed1,devsb,heldout,ir}/`; `calib/`; `probe_krea/`; `logs/task_<name>.log`, `logs/fleet_<jobid>.out`; `telemetry/`.
- Old FLUX.1-dev outputs in `out/` are NOT for any paper (lattice regime).
Prompts: `prompts/aesthetic4k_eval.jsonl` (195 Aesthetic-4K eval captions, refs in aesthetic4k_refs.jsonl),
`prompts/calib_pickapic.jsonl` (39 filtered, calibration only).

## 5. Original protocol (completed campaigns)
Dev = eval prompts 0-7 (choose variants/hyperparameters here only). Held-out = prompts 8-194 (report only these).
The already inspected held-out set is not fresh confirmation for new variants. Third-diagnostic selection uses the separate frozen64 confirmation plan.
Main baselines already exist for held-out: ntk2, dype, sega_official (out_krea/main), temp_only_sbase and constancy_tm_sbase
(out_krea/mainsb). Any new method: dev run (8 prompts, ~30 min/GPU) -> if it beats sega_official on dev on IR AND MUSIQ, with CLIP delta >= -0.3; select at most two ->
held-out main (27 chunks of 7 prompts, ~25 min each) -> `scripts/heldout_analysis.py` (add the new root to the loader).

## 6. Historical second-campaign execution notes (superseded for new work)
- Task script template: copy any `taskq/done/52_c*_main_ctmsb.sh` and edit method/out/flags; `bash -n` it; drop in taskq/pending.
- The completed second campaign ran on externally launched fleet 1076092 on c672-016 (four GPUs, gb/CR13 configuration).
  The originally recorded CR10 job has no visible log. The previous monitor-only session forbade SSH, sbatch, and scancel. That session restriction is historical; the current user explicitly authorized full project operations. Preserve this completed campaign.
- Eval: copy `taskq/done/60_eval_mainsb.sh` (root/out/reference) and `61_ir_main.sh`.
- When the campaign is complete and pending is empty, touch taskq/STOP so the existing workers exit.
  Do not cancel jobs. Do not modify worker_loop.sh or node_launch.sh.

## 7. Completed second-campaign archive
The authorized B1/B3 campaign is complete. Its original spool remains stopped. New work uses taskq_dev3.
1. Read `.agent/notes/main2/REPORT.md` for the dev decision and full held-out result.
2. Dev sources/decision live in `.agent/notes/dev2`; held-out means, paired CIs, per-prompt data,
   prompt manifest, and final raw scorer reports live in `.agent/notes/main2` on work.
3. Full generated images remain at out_krea/main2/sega_ctm/4096x4096 on scratch. All 188 passed the
   generation audit; held-out analysis excludes row 7 and uses exactly 187 paired seed-0 prompts.
4. All 53_ and 62_ tasks succeeded. STOP was touched at 18:05:35 CDT after pending/running were empty.
   All four workers exited by 18:06:06; no scheduler cancellation occurred.
   Do not submit or rerun anything for this completed campaign. Preserve the negative evidence.
5. Both aliases and eta 0.5 stopped at dev. Do not reinterpret dev scores as results or tune on held-out.
Formulas and frozen rule: `.agent/notes/alias_temp.md`; task/source hashes: `.agent/notes/dev2_manifest.json`.
New modes and analysis plumbing were subsequently committed as a1ede1a.
CPU-only prepasses used local receipts after a documented W&B socket failure. Final GPU evaluation
logged offline W&B run n8xhe6iy; no remote synchronization is claimed. No worker/node-launch edits,
SSH, Slurm submission/cancellation, package installs, or scratch deletion occurred in this session.

## 8. Process rules for any agent here
Vista: compute only on compute nodes; login node only for sbatch/squeue/short ssh, max 3 concurrent ssh, never ssh in loops.
Never pkill/killall; kill only exact PIDs you created. uv only. Never delete scratch data (move aside instead).
No em dashes or section symbol in prose. Commits: author shreshthsaini, no Co-Authored-By/Claude trailers.
Update `.agent/STATE.md` and this file after every meaningful step.

## Operator log (Claude/Fable; Codex6 owns .agent/STATE.md during the campaign, so paper-side status lives here)
- 2026-10-19 03:35: paper v2 camera-ready committed (3211453), paper/dist PDF + zip sent to user. Body 9 pages, 25 total.
- 2026-10-19 04:00: user instruction: figures/diagrams/plots must follow the reference papers we cite (Toronto SEGA/AReA,
  looped diffusion transformer paper, the "budget" paper, ByteDance, DeepMind), with minimal text inside figures. In progress:
  Opus max design agent writes paper/figs/STYLE_REFERENCE.md and paper/figs/CRITIQUE_v2.md; then Codex implements the figure
  changes; then Claude reviews every rendered page and sends v3. Memory rule saved: paper-figures-professional-bar.
- Strengthening campaign: 315 tasks queued; fleets CR22/CR23/CR24 pending; Codex6 polls and will score + draft the paper
  update; Claude verifies before any new dist is sent.
- 2026-10-19 18:50: session model Opus 5.5. User: no idle GPUs/nodes, parallelize, finish fast; Opus and Fable in background.
  Fleets: CR22 1080046 (gh 8n 24h, est 18:50), CR25 1082718 (gh 8n 12h), CR23 1080434 (gb 1n 12h), CR24 1080435 (gh 2n 8h),
  all draining taskq (315 tasks). Event monitor: slurm/utilization_watch.sh (FLEET/FAILED/UNDERUSED/QUEUE_EMPTY/IDLE_OTHER).
  Gold-standard bar: ~/.claude/skills/gold-standard-paper/SKILL.md (binding). Two Opus max agents running: figures agent owns
  paper/figs (redesign to gold recipe, minimal text, PALETTE.md); text agent owns main.tex/sec/tables (abstract, intro,
  algorithm pair, tables with Key result, wrapfigures, appendix TOC, rubric self-score here). Next: Fable rubric review, then
  refresh paper/dist and send v3. Codex6 later adds confirmation results to the paper.
- 2026-10-19 23:40: CR22 RUNNING since 21:26 (8 GH200): factorial 75_ nearly done (about 140 images per arm, configs verified:
  ntk2_sref = ntk2 base + sref temp; sega_base_none = SEGA base, no temp; guidance 4.5, seed 0). Codex6 DIED: OpenAI Codex usage
  quota exhausted until 2026-10-25 04:01; it never published the 90_ eval/IR tasks or the 70_/82_ Qwen tasks (queue holds only
  generation). Opus max analyst agent now owns: publishing evals per block, analysis (scripts/strengthen_analysis.py,
  eval_krea/strengthen/, .agent/notes/strengthen/RESULTS.md, STATUS.md), and drop-in paper files (sec/results_confirmation.tex,
  tables/strengthen_tables.py, figs/strengthen/, paper/INTEGRATION.md). Figure and text Opus agents resumed after rate limits.
- 2026-10-19 23:45: figures v3 committed (gold recipe; words inside figures: F1 17, F2 17, F3 29, F4 11, F5 16; palette in
  figs/PALETTE.md; captions + LaTeX handoff in figs/CAPTIONS.md). Reviewed by Claude: pass. Text agent integrating the handoff.
  10 idle GPU nodes in the user's other jobs; guest launch blocked by permission classifier; user can run
  `bash slurm/launch_guests.sh` to use them.
- 2026-10-20 00:05: PUBLIC REPO https://github.com/shreshthsaini/constancy-rope (origin, branch main; user asked to upload
  everything). Push after every meaningful commit (`git push`). Build PDFs, paper/dist and superseded heldout_* figures are
  untracked (.gitignore). Results stay on scratch; paper crux lives in paper/tables and paper/figs scripts + data pointers.
- 2026-10-24 07:00: scoring tasks (22 x 90_) published for factorial, tsweep, fresh, seeds, ckpt_matrix, default_shift; waiting
  for fleet CR26 1096129 (gh 3n 12h, pending since 00:45). Checkpoint x guidance control DONE: FLUX.1-dev lattice power 12.0x
  [7.8,18.6] Krea at guidance 3.5 and 7.4x [4.5,12.7] at 4.5; guidance change scales dev lattice 0.52x, Krea 0.89x: the
  lattice is a checkpoint effect. Qwen NOT RUN (OOM from a co-tenant 45 GiB process per GPU; retry needs ~75 GPU-h, not
  affordable at 2,803 SU). DO NOT EDIT scripts/strengthen_analysis.py, scripts/eval.py, crope/metrics before the 90_ tasks
  run (checksum-gated). AFTER THE FLEET: `crope_py scripts/strengthen_confirm.py run`, then `crope_py
  paper/tables/strengthen_tables.py`, then the fs_*.py scripts in paper/figs/strengthen/; the drop-in section
  paper/sec/results_confirmation.tex prints each paragraph once its block is analysed; paper/INTEGRATION.md says where to
  \input it. Text fix pass (before/after numbers, float order, rubric 1.2) was stopped on 10-20; relaunch when results land.
