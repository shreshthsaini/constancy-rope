# Results log (rewrite sections, keep short)

## Smoke (2026-10-15)
1024^2 9.2 s/img; 4096^2 291 s/img (probe runs ~330 s), peak 40.8 GB, GPU util 100% at ~810 W (compute-bound, batching would not help). Sigmas correct for mu=1.15.

## T2 probe, prompt 0, seed 0, gain x1.5 per octave (probe/4096x4096/mtf.csv)
Elasticity d log P_out / d log g_in. Diagonal latent height [0.19,0.05,0.25,0.22,-0.64,-3.18], width [1.47,-2.79,0.55,-3.05,-1.90,-3.64].
Highest band (0.125-0.25 cyc/token) gain up -> power DOWN at high bins on both axes (~ -3.6, i.e. x0.23); image domain agrees.
Single prompt; perturbation changes trajectory so noise is large. Extra probes queued: 35_probe_extra_{p1,p2,f067} (prompts 1,2 at x1.5; prompt 0 at x0.667).
Implication: the pre-registered sign (deficit -> raise gain) may be wrong for high bands. Decide with pilot metrics + extra probes, report honestly.

## Pilot images (2026-10-15 12:30) -> PIPELINE OR MODEL PROBLEM
Native 1024 base is clean. At 4096^2 with FLUX.1-dev: ntk2 = featureless blur; dype and constancy = dense dot/halftone lattice over whole frame (pilotviz sheets in session scratchpad). Not publishable; pilot + probe numbers NOT trusted until resolved.
Official DyPE and SEGA both use FLUX.1-Krea-dev (DyPE run script guidance 4.5). Diagnostic 05_diag_official_dype_dev runs official DyPE code with our FLUX.1-dev at 4096 (diag/). Krea transformer downloading (logs/download_krea.log).
Decision rule: official-DyPE+dev also lattice -> checkpoint cause, switch campaign to Krea (matches both baselines' papers). Official clean -> bug in crope rope/gain path, diff against official.

## Checkpoint diagnosis (2026-10-15 13:10) -> SWITCH TO KREA
Official DyPE code + FLUX.1-dev at 4096^2 (guidance 3.5): same dense lattice texture as ours -> checkpoint cause, not our code.
Official DyPE + FLUX.1-Krea-dev (guidance 4.5): clean composition at 4096^2 (diag/official_dype_krea_4096_g4.5_dype1.png), 192 s.
Campaign now Krea + guidance 4.5 (matches DyPE and SEGA official). Roots: out_krea, calib/krea_profile.npz, probe_krea, eval_krea.
The FLUX.1-dev T2 probe result above is void (lattice regime). Krea probe: 20_probe (prompt 0) + 35_probe_krea_p1/p2.

## T2 probe on Krea, prompt 0 (probe_krea/4096x4096/mtf.csv)
Diagonal elasticity vs mean off-diagonal: latent height [0.36,0.10,-0.08,0.08,0.34,0.08] vs 0.06; latent width [1.22,0.73,0.08,0.36,0.30,0.11] vs 0.18; image height [0.32,0.08,-0.13,-0.10,0.47,-0.02] vs 0.11; image width [1.69,0.78,0.32,0.49,0.51,0.20] vs 0.20.
Coupling positive and band-selective on width, weak on height. Supports the method's sign assumption (FLUX.1-dev negative result was the lattice). Prompts 1, 2 pending.

## Krea pilot timing
~290 s/img for ntk2, sega_official, constancy; dype 330 s. Official DyPE script 192 s (pipeline overhead in ours to investigate).

## Pilot eval, Krea, 8 prompts 4096^2 (eval_krea/pilot/4096x4096/summary.csv), 2026-10-15 17:30
clip/pick/musiq/musiq_crop/slope_err: constancy 30.10/23.10/40.6/58.6/0.174; sega_official 30.32/23.41/42.5/58.2/0.099; temp_only 29.96/23.12/40.4/60.3/0.182; dype 28.59/22.23/44.4/58.3/0.150; ntk2 27.84/22.37/32.9/41.4/0.132; yarn 28.06/21.14/42.5/66.7/0.240; paper sega 28.14/22.18/30.8; sega_static (paper) broken (clip 13.1); sega_official_static ~= sega_official.
Diagnosis: constancy ~= temp_only. Per-band correction is exactly 0 for steps 0-23 and acts only at sigma<=0.35 (log g change -0.1..-0.6 on high pairs). Constancy shows high-freq excess in decoded images vs A4K refs (spectrum_log_error b07-b09 +0.75..+1.35).
Missing metrics: ImageReward (crope transformers too new -> scripts/imagereward_score.py in mend env), FID/KID (torch-fidelity installed now), HPSv3 skipped.
Protocol decision: prompts 0-7 are the DEV set for eta/gate choice; final numbers on held-out prompts 8-194 only. Sweep queued: 12_sweep_{constancy_eta2,constancy_eta4,constancy_nogate}. Constancy main (eta=1) tasks held in taskq/hold (7 chunks already ran at eta=1, keep as ablation).

## Dev time-matched (2026-10-15 19:48), prompts 0-7
IR / pick / musiq / spec_rmse: sega_official 1.423/23.41/42.5/0.857; constancy_tm eta1 1.314/23.10/41.9/0.998; eta2 1.313/23.04/43.3/1.018; constancy 1.285; temp_only 1.282. Correction now active from step ~8 (pairs 6-8 lowered up to -0.57). Still below sega_official.
Confound: ours use ntk2 base + tau_N (1.333 logit); sega_official uses its own NTK base and m_ref (~1.25 logit). Controlled dev runs on SEGA base + sref temp: 11_devsb_temp (no allocation), 11_devsb_ctm (constancy_tm allocation); chain logs/devsb_chain.sh scores them vs sega_official (its own allocation).

## CONTROLLED dev (SEGA RoPE base + sref temperature), prompts 0-7 (eval_krea/devsb), 2026-10-15 20:54
IR/pick/clip/musiq: no allocation 1.444/23.26/30.32/42.7; constancy_tm allocation 1.432/23.25/30.24/41.7; SEGA allocation (sega_official) 1.423/23.41/30.32/42.5. All within dev noise.
=> Per-band allocation (SEGA's or ours) adds nothing measurable; RoPE base + global temperature drive the gains. Constancy does NOT beat released SEGA (ties). Do not claim a win.
Held-out confirmation queued: 52_cXXXXX_main_{tsb,ctmsb} (temp_only and constancy_tm on CROPE_ROPE=sega_official, --temperature sref), out_krea/mainsb/, prompts 7-194 (report 8-194). Constancy eta=1 ntk2-base main superseded (7 chunks done, rest left in taskq/hold).
Paper decision: user said paper only if it beats baselines. Present honest status; the defensible paper is an analysis ("what drives training-free RoPE extrapolation"), pending held-out CIs.

## HELD-OUT RESULT (prompts 8-194, n=187, 4096^2, Krea, seed 0) -- eval_krea/heldout/, 2026-10-16 04:05
Paired vs sega_official (bootstrap 95% CI):
- constancy_tm_sbase: IR +0.018 [-0.023,0.059] tie; pick -0.046 tie; clip -0.27 worse; musiq -3.0 worse; musiq_crop -2.2 worse; clipiqa -0.025 worse; spec rmse +0.019 worse; slope err +0.018 worse; tiling -0.007 better.
- temp_only_sbase (no allocation): IR/pick/clip/musiq_crop tie; musiq -2.1 worse; clipiqa -0.009 worse; spectral slightly worse; tiling better.
- dype: IR -0.16, pick -0.49 worse; musiq_crop +2.0, clipiqa +0.03 better. ntk2 far worse.
VERDICT: Constancy RoPE (best variant) does NOT beat released SEGA; it is worse on most image-quality metrics and slightly worse than no allocation. SEGA's released allocation gives a small real full-res MUSIQ/CLIP-IQA gain over no allocation (dev-set claim "allocation adds nothing" was wrong on MUSIQ). Method hypothesis refuted. No method paper.
