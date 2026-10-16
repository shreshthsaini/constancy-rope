# Constancy RoPE: handover (read this first)

Last full rewrite: 2026-10-16 05:10 CDT. Live short state: `.agent/STATE.md`. Result log: `.agent/notes/results.md`.
Design spec: `.agent/notes/design.md`. Codex briefs: `.agent/notes/codex_brief_{1,2,3}.md`. Review of the SEGA paper that
started this: `../research_reviews/sega_2605.22668_2026-10-14.md` (+ artifact https://claude.ai/artifact/Phv2pGDvZDZwpmNM3Qh8GL).

## 1. Goal and the user's rules
- Training-free resolution extrapolation for FLUX at 4096^2. Beat released SEGA (arXiv 2605.22668, github rajabi2001/sega)
  with a principled, vision-science-derived per-band RoPE attention gain ("Constancy RoPE").
- User: write a paper ONLY if results are legit and the method beats the baselines (close to SOTA is fine if it beats them).
  Never report dev-set wins as results. Keep this file detailed so a fresh agent can continue after a crash.
- User authorized holding GPUs on any partition and using Codex in the background.

## 2. Verdict so far (do not relitigate without new evidence)
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
=> Method hypothesis refuted as designed. Paper draft in `paper/` assumes a win: rewrite before any use.

## 3. Hard-won facts
- FLUX.1-dev at 4096^2 gives a dot/halftone lattice even under OFFICIAL DyPE code -> use FLUX.1-Krea-dev (both DyPE and SEGA do).
  Loader: env `CROPE_TRANSFORMER=<Krea snapshot>` (Krea transformer + FLUX.1-dev text encoders/VAE). Task scripts export it.
- diffusers default mu extrapolates to ~11.5 at 4K and collapses sigmas; official repos pass max_seq_len = image_seq_len
  (mu = 1.15). Patched in `crope/runtime.py` (register_to_config before configs are recorded).
- Released SEGA code differs from its paper (alloc 1 - 0.15*spread*tanh(1.5 z), clipped >= 1, x_embedder features,
  NTK base / (1 + 0.1 ln s)). `sega_official` = released; `sega` = paper equations. See `crope/THIRD_PARTY.md`.
- Band power must be a grid-independent density (orthonormal FFT, mean per band). Fixed.
- SDPA: prefer cuDNN on GH200 (215 s/img at 4096^2 vs 290 s with forced flash). Peak 41 GB. GPU compute-bound at 100%.
- Krea native 1024^2: 5.5 s/img. Calibration (16 Pick-a-Pic prompts, disjoint from eval) -> `calib/krea_profile.npz`,
  time-resolved `calib/krea_profile_tm.npz`.
- ImageReward needs the `mend` env (crope's transformers too new): `scripts/imagereward_score.py`.
- `CROPE_ROPE=<rope method>` env runs any controller on another method's RoPE base (recorded in config["rope"]).
- T2 probe on Krea: raising a band's gain raises that band's power on the width axis (elasticity 0.1 to 1.7), weak on height.

## 4. Layout
Code (git, local only, no remote): `/work/09032/saini_2/vista/Projects/Exploration/constancy_rope`
- `crope/` runtime.py (pipeline, FluxExperiment, resume), gain.py (controllers + attention processor), rope.py (pos-embed
  variants), spectral.py, sega_official.py, metrics/. `scripts/` sample.py, calibrate.py, probe_mtf.py, eval.py, enqueue.py,
  imagereward_score.py, heldout_analysis.py. `tests/` (156 pass: `OMP_NUM_THREADS=8 <crope python> -m pytest -q tests`).
- `slurm/` fleet_CR*.sbatch, node_launch.sh (per-GPU workers; kills telemetry after workers exit), worker_loop.sh
  (atomic claim, requeue on SIGTERM, exits after 90 min idle or when taskq/STOP exists).
Env: `/work/09032/saini_2/vista/library/uv-envs/crope/bin/python` (torch 2.11 cu128, diffusers 0.39). ImageReward: `.../uv-envs/mend/bin/python`.
Scratch: `/scratch/09032/saini_2/constancy_rope/`
- `taskq/{pending,running,done,failed,hold}` + `STOP` marker (PRESENT now; remove before starting a new fleet).
- `out_krea/{pilot,main,mainsb,seed1,devtm_eta*,devsb,sweep}/METHOD/4096x4096/*.png+json`; `eval_krea/{pilot,main,mainsb,
  seed1,devsb,heldout,ir}/`; `calib/`; `probe_krea/`; `logs/task_<name>.log`, `logs/fleet_<jobid>.out`; `telemetry/`.
- Old FLUX.1-dev outputs in `out/` are NOT for any paper (lattice regime).
Prompts: `prompts/aesthetic4k_eval.jsonl` (195 Aesthetic-4K eval captions, refs in aesthetic4k_refs.jsonl),
`prompts/calib_pickapic.jsonl` (39 filtered, calibration only).

## 5. Protocol (keep it)
Dev = eval prompts 0-7 (choose variants/hyperparameters here only). Held-out = prompts 8-194 (report only these).
Main baselines already exist for held-out: ntk2, dype, sega_official (out_krea/main), temp_only_sbase and constancy_tm_sbase
(out_krea/mainsb). Any new method: dev run (8 prompts, ~30 min/GPU) -> if it beats sega_official on dev on IR AND MUSIQ ->
held-out main (27 chunks of 7 prompts, ~25 min each) -> `scripts/heldout_analysis.py` (add the new root to the loader).

## 6. How to run things
- Task script template: copy any `taskq/done/52_c*_main_ctmsb.sh` and edit method/out/flags; `bash -n` it; drop in taskq/pending.
- Submit a fleet (from this compute node): check `ssh login1 /usr/local/etc/taccinfo` (stop at 96% burn), then ONE
  `ssh login1 'sbatch /work/.../slurm/fleet_CR3.sbatch'`. Account ASC26048. Never 1-node-per-task arrays. gb QOS allows 3 submits.
  User's account often near the 40-job submit cap (other sessions). gh-dev: 1 job/user.
- Eval: copy `taskq/done/60_eval_mainsb.sh` (root/out/reference) and `61_ir_main.sh`.
- Release nodes when nothing runnable remains: touch taskq/STOP; if jobs linger, scancel the exact CR job ids.

## 7. Open directions (user decides; Codex may pursue option B)
A. Analysis paper ("what drives training-free RoPE extrapolation"): controlled ablations above, checkpoint dependence,
   SEGA paper-vs-code gap, temperature as the main lever. All data exist.
B. New method iteration, same protocol. Evidence says the lever is the attention temperature and RoPE base, and SEGA's
   released allocation adds a small sharpness gain. Candidates: (B1) per-band entropy-matched temperature calibrated against
   the native model at the same sigma (adaptation / divisive normalization), (B2) derived RoPE base from the native
   positional-spectrum match, (B3) SEGA allocation + time-matched target as a correction on top of SEGA (not replacing it).

## 8. Process rules for any agent here
Vista: compute only on compute nodes; login node only for sbatch/squeue/short ssh, max 3 concurrent ssh, never ssh in loops.
Never pkill/killall; kill only exact PIDs you created. uv only. Never delete scratch data (move aside instead).
No em dashes or section symbol in prose. Commits: author shreshthsaini, no Co-Authored-By/Claude trailers.
Update `.agent/STATE.md` and this file after every meaningful step.
