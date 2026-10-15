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
