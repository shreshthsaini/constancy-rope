You are continuing a research campaign autonomously. FIRST read, in order: HANDOVER.md, .agent/STATE.md, .agent/notes/results.md
(all in /work/09032/saini_2/vista/Projects/Exploration/constancy_rope). They explain the method, the infrastructure, what failed
and why. The previous method (Constancy RoPE) does not beat released SEGA on held-out data. Your job: implement and test two new
candidates under the SAME dev/held-out protocol, honestly, and keep the handover files current.

CANDIDATES (both training-free, few lines, run on SEGA's RoPE base via CROPE_ROPE=sega_official):
B1 "alias_temp" (per-band entropy-invariant temperature). Jin et al. 2023 keep attention entropy fixed with a logit gain
   ln N_tgt / ln N_train because the softmax pool grows from N_train to N_tgt keys. Per RoPE band this pool is not the whole
   grid: a band of wavelength lam (tokens, after the RoPE base change) cannot tell apart keys that differ by whole periods,
   so the number of positions it distinguishes along an axis of length L is n(lam, L) = min(L, max(1, L/lam)) per axis
   (a grid-cell aliasing count; the 2D count is the product over the axes this pair encodes, i.e. 1D per axis here since
   Flux RoPE is axial). Logit gain for pair d on axis a: g_d = ln(n_tgt,d * N_other_tgt) / ln(n_nat,d * N_other_nat), where
   N_other is the token count along the other axis (the keys this pair cannot separate are still in the pool). Make the
   exact formula explicit in code comments, derive it in .agent/notes/alias_temp.md, and test limiting cases: lam -> 0 gives
   ln N_tgt/ln N_nat (Jin et al.); lam >= L_tgt gives the ratio for the other axis only. Axis-0 (text index) dims: global
   Jin ratio. Also a variant "alias_temp_sref" that rescales all g_d so their geometric mean equals SEGA's reference gain
   (s^0.16), to separate allocation from global level.
B3 "sega_ctm" = released SEGA allocation (mode sega_official) multiplied by the time-matched constancy correction
   exp(eta * w_b * clip(e_b,-1,1)) from constancy_tm (no extra temperature; eta default 0.5). This tests whether the
   vision-derived correction helps as a refinement ON TOP of SEGA rather than as a replacement.

WORK PLAN
1. Implement modes in crope/gain.py (+ runtime/sample/enqueue plumbing), unit tests, full suite must pass
   (OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python -m pytest -q tests). You cannot git commit
   (read-only .git); leave changes in the tree and list them in your report.
2. Dev runs on eval prompts 0-7 at 4096x4096 for: alias_temp, alias_temp_sref, sega_ctm (eta 0.5) and sega_ctm (eta 1.0).
   Write one task script per run into /scratch/09032/saini_2/constancy_rope/taskq/pending/ named 13_dev_<variant>.sh, copied
   from taskq/done/11_devsb_ctm.sh (keep its exports: CROPE_TRANSFORMER, CROPE_ROPE=sega_official, HF_HOME etc), outputs to
   /scratch/09032/saini_2/constancy_rope/out_krea/dev2_<variant>/<method>/4096x4096. `bash -n` every script. A GPU fleet
   (job CR10) drains taskq; it may still be queued: just wait (poll every 5 min with sleep; never busy-loop).
3. When dev images exist: run eval for each dev root (copy taskq/done/42_eval_devsb.sh into pending as 43_eval_dev2_<v>.sh with
   --root/--out changed) and ImageReward (copy taskq/done/61_ir_mainsb.sh as 43_ir_dev2_<v>.sh with root/csv changed).
   Compare against sega_official on the same 8 prompts (eval_krea/pilot/4096x4096/per_image.csv, eval_krea/ir/pilot.csv).
4. Decision rule (pre-registered): a variant goes to held-out only if on dev it beats sega_official on BOTH ImageReward and
   full-res MUSIQ means, and is not worse on CLIP by more than 0.3. Pick at most 2 variants. Held-out = prompts 7-194 in 27
   chunks of 7 (copy the structure of taskq/done/52_c*_main_ctmsb.sh), outputs out_krea/main2/<method>/4096x4096, prefix 53_.
   Then eval + ImageReward on out_krea/main2 (prefix 62_), then extend scripts/heldout_analysis.py to load main2 (suffix
   _v2) and run it. Report paired deltas vs sega_official on prompts 8-194 with CIs.
5. If no variant passes dev, stop GPU work: write the negative result clearly, touch taskq/STOP only if taskq/pending is empty.

AFTER EVERY STEP update .agent/STATE.md (rewrite the Status section, keep under 120 lines) and append findings to
.agent/notes/results.md, and keep HANDOVER.md section 2 (verdict) and 7 (directions) accurate. A fresh agent must be able to
resume from those files alone.

HARD CONSTRAINTS
- You are on a shared CPU compute node (i618-122). OMP_NUM_THREADS=8, keep your own processes under 20. No GPU here.
- NEVER run ssh, sbatch, scancel, squeue-based kills, pkill or killall. Do not touch other projects. Do not modify
  slurm/worker_loop.sh or slurm/node_launch.sh. Never delete anything under /scratch (move aside if needed).
- Env: crope python above (install nothing into it; ImageReward runs via the mend env as the 61_ir task does).
- Honest reporting: dev numbers are for selection only; never present them as results. No em dashes or section symbol.
- Done = either held-out results for passing variants with the analysis table, or a documented negative dev result.
  Final report: what you implemented, dev table, decision, held-out table if any, files changed.
