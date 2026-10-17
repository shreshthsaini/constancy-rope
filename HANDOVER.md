# Constancy RoPE: handover (read this first)

Last takeover update: 2026-10-17 CDT. Live short state: `.agent/STATE.md`. Result log: `.agent/notes/results.md`.
Design spec: `.agent/notes/design.md`. Codex briefs: `.agent/notes/codex_brief_{1,2,3}.md`. Review of the SEGA paper that
started this: `../research_reviews/sega_2605.22668_2026-10-14.md` (+ artifact https://claude.ai/artifact/Phv2pGDvZDZwpmNM3Qh8GL).

## Current takeover checkpoint (2026-10-17)
- User requested unattended continuation through the whole research project. Watcher `scripts/watch_cr14.py` v2 is running in compute-node tmux `crope-watch-cr14`, PID1105783 on i618-051, controller job1073260. It directly starts this Codex thread when idle through the local app-server socket; active turns retain pending events. No model calls while waiting or SSH. Scheduler120s/files60s; allocation, workflow results, failure/inactivity, report, and controller-expiry events. It remains active after the CR14 report. Nine tests passed; live thread/cwd and busy-turn deferral verified. The direct WAKE_TEST is armed: next automatically started turn must acknowledge actual receipt in `.agent/notes/watcher_ack.json`, then continue the same-state response probe from `response_probe_plan.json`. See `.agent/notes/watcher_cr14.json` and live scratch state for lease, heartbeat, and stop instructions. Every next campaign must be registered with the watcher before waiting; migrate before the controller allocation expires.
- Current success target: a genuinely novel mechanism, distinct from recent methods, with close-to-or-better-than-SOTA performance in matched training-free DiT resolution extrapolation. Neither requirement is currently met. See `.agent/notes/novelty_20261017.md`.
- User first requested a state review, then explicitly assigned full project ownership and
  authorized continuing experiments, evaluation, and paper work autonomously.
- User explicitly authorizes background Codex agents, GPT-6 Astra, and GPT-6.1 Sol with effort chosen by the main agent; use higher effort for critical work. Respect tool concurrency and login-guard limits.
- Third diagnostic submitted as CR14 job1076857, currently PENDING Priority on two GH200 nodes.
  See `.agent/notes/continuation3/PROTOCOL.md`; frozen source commit395ba85, 256 CPU tests passed, including 19 confirmation checks.
  Three parameter-free/timing controls plus fresh SEGA, 32 development images, no held-out tuning.
- Live accounting confirms CR13 job 1076092 COMPLETED October 16 at 18:06:13 CDT.
  Spool: pending 0, running 0, done 255, failed 5, hold 25; STOP remains present.
  Failed/held files are older pilot, superseded original-constancy, or archived task files.
  CR14 is the only newly submitted project job; unrelated allocations were left unchanged.
- Last research commit: a1ede1a, October 16 at 18:11:57 CDT. Earlier uncommitted-state notes
  are superseded. Repository has no configured remote.
- `paper/main.tex` now builds an empirical working report from `sec/current_report.tex` and
  `sec/current_table.tex`. Earlier proposal sections remain in source but are excluded until audited.
  PDF: `paper/constancy_rope_working_2026-10-17.pdf`. No positive efficacy claim is supported.
- Added fixed-order qualitative examples from held-out rows 8-10, chosen before image viewing. Six source PNG hashes and sidecars verified; native center crops and display assets archived in `paper/figs/heldout_example_assets/`. PDF figures have clean builds and visual inspection.
- `paper/update_evidence.py` reproduced all ten selected-candidate paired means/CIs exactly
  from archived per-prompt CSVs. Receipt: `.agent/notes/paper_verification_20261017.json`.
- Current step: monitor job1076857, verify telemetry and source/protocol invariants, then
  run `scripts/dev3_analysis.py` with `--reference-root dev3_sega_official`. No GPU outcome yet.
  Fresh64 confirmation manifest remains unused; a dev-control pass alone does not authorize advancing it as the main method. Require a distinct mechanism and current-baseline protocol.
  The last successful worker host runs `scripts/dev3_finalize.py`: validates all four outputs,
  archives the selection, logs W&B, rebuilds the PDF, and commits its own outputs/state.
  It never treats a dev pass as a held-out result or submits confirmation automatically.
- Prepared confirmation generation and analysis with immutable prompt/controller checks, paired bootstrap rules, and hardware receipts. No confirmation tasks submitted. See `confirmation_plan.md` and `confirmation_implementation.json` under `.agent/notes/continuation3/`.

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
