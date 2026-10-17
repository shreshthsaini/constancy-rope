# Constancy RoPE: handover (read this first)

Last takeover update: 2026-10-17 CDT. Live short state: `.agent/STATE.md`. Result log: `.agent/notes/results.md`.
Design spec: `.agent/notes/design.md`. Codex briefs: `.agent/notes/codex_brief_{1,2,3}.md`. Review of the SEGA paper that
started this: `../research_reviews/sega_2605.22668_2026-10-14.md` (+ artifact https://claude.ai/artifact/Phv2pGDvZDZwpmNM3Qh8GL).

## Current takeover checkpoint (2026-10-17, 08:43 CDT)

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
- CR18 **job1077211 is PENDING Priority**, two GH200 nodes/15min. Separate two-prompt pretrained BF16 audit, ten denoiser forwards including no-op repeats and four-arm mediation. Artificial sigma0.5 states mix validated CR17 endpoints with fixed CPU noise. Source061fa03; frozen `.agent/notes/observer_pretrained_protocol.json` SHA b98de615bdb3c475b38f355f35b13fbce50240f614aaac9492ebf803a77374da.23CPU tests pass, independent code/protocol review found no blocker, final accepted-input preflight passed. No natural-trajectory, quality, novelty or confirmation claim. Submission receipt: `.agent/notes/observer_pretrained_submission.json`. Existing watcher registry atomically includes the new queue/progress/completion paths; hot-read verified; watcher scheduler observation verified.
- CR15 job1077082 remains FAILED under its original protocol: both2048 endpoints matched, no-op replay was exact, but histories differed; no4K stage started. Archive: `.agent/notes/response_cr15_failure/`. Do not reclassify it.
- CR16 job1077122 completed a separate repeatability diagnosis. Ordinary baselines also drifted in history, while observed BF16 gains, state/velocity hashes and endpoints matched. The largest FP32 gain difference was1.19e-7. Eight isolated deterministic allocation repeats matched for each prompt. This localizes the issue to feature-to-gain arithmetic, not a unique kernel. Archive: `.agent/notes/response_cr16_repeatability/`.
- CR17's separate frozen source1a06786 and `response_deterministic_protocol.json` declare deterministic algorithms only inside OfficialSEGA.update, with flags restored afterward. No general quality equivalence with the original numerical policy is established. CR14 and original CR15 sources remain unchanged.
- One watcher runs in tmux `crope-watch-cr14`, PID1581602, on i618-051/controllerjob1073260. Runtime: `/scratch/09032/saini_2/constancy_rope/watcher_cr14`. Hot registry includes CR15/16/17/18; CR14 is also watched. Direct autonomous wake was verified; receipt `.agent/notes/watcher_ack.json`. Never duplicate the watcher. Register new jobs before waiting.
- Controller lease ends **12:45:04 CDT**. Watcher alerts one hour before expiry. A future compute-node handoff still requires execution and verification. Latest budget check was7941/40000SU remaining at08:28CDT; refresh before each submission and stop requesting nodes at96 percent burn.
- Current paper: `paper/constancy_rope_working_2026-10-17.pdf`, eight-page empirical report. CR14 results and qualified CPU observer audits are included, build clean, pages6/7 visually checked. DoPE prior-art boundary added, pages3/8 visually checked after clean rebuild. Receipt: `.agent/notes/observer_paper_receipt.json`. No SOTA claim. Prior held-out numbers and fixed-order qualitative examples are preserved.
- Post-observer research review rejects covariance-aware gain fitting without an independently justified correction target. DoPE is an additional verified rotary-band Gram-spectrum prior-art collision. Decision: `.agent/notes/post_observer_research_decision.json`. No new GPU sweep or fresh64 use. A separate resolution-focused pivot review is in progress.
- All eight CR14 baseline development images were visually inspected in fixed order at768px; no obvious global duplication was seen at that scale. Row5 requested shop text is garbled, but no resolution-specific cause is established. Receipt: `.agent/notes/cr14_visual_review.json`. Do not turn this exploratory observation into a quality or causal claim.
- CPU CR18 report helper is implemented and under independent review, separate from frozen runtime. Archive accepted JSON on work. If CR18 enters the paper, also preserve saved input/output/observer/target tensors outside the implementation repo on work with a hash-mapping receipt; these modest-size tensors are the recomputation witnesses.
- Prior-art notes: `response_prior_art_20261017.md`, `response_actuator_prior_art_20261017.md`, and `novelty_20261017.md`. AFM already edits attention Fourier bands; finite differences, Jacobian guidance, normalization, inversion and reuse are established. A local response map alone does not establish a new generation method.
- The first two campaigns are complete and negative; their inspected187 held-out prompts cannot become fresh confirmation. Old taskq remains stopped with255 done,5 historical failures and25 held. Do not revive it. Fresh64 remains unused and needs a distinct method plus a recent-baseline protocol before confirmation.
- Repository is local-only with no configured remote. Latest completed evidence commits:90cc4c7 and3141476. Keep implementation and paper-critical crux on work; bulk artifacts remain on scratch.

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
