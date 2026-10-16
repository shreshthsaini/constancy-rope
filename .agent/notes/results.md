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

## Seed-1 replication (prompts 8-62, n=55), paired vs sega_official
temp_only_sbase: IR +0.031 tie, pick tie, clip tie, musiq -2.79 worse, musiq_crop tie, clipiqa tie.
constancy_tm_sbase: IR +0.019 tie, pick tie, clip tie, musiq -3.53 worse, musiq_crop -1.99 worse, clipiqa -0.025 worse.
Same pattern as seed 0. Verdict stable. Workers stopped (taskq/STOP) and CR8 cancelled 2026-10-16 04:45: no runnable work left.

## Ops lesson (2026-10-16)
Fleet jobs did not end after workers exited: the background gpu_telemetry.sh kept the srun step alive. Fixed in slurm/node_launch.sh (kill telemetry after workers). CR3/CR4 scancelled 04:52 with no runnable work left.

## Second campaign: implementation and dev queued (2026-10-16 11:20 CDT)
Implemented alias_temp, alias_temp_sref, and sega_ctm (default eta 0.5, also testing 1.0). Exact formulas, assumptions, limiting cases, and selection rule are in alias_temp.md. All run on released SEGA RoPE. B3 composes released projected-feature gain with only the time-matched correction. Full CPU suite: 171 passed in 13.75s. Four 13_dev_*.sh tasks copied from 11_devsb_ctm.sh, bash -n passed, outputs isolated under out_krea/dev2_<variant>. Manifest and source hashes in dev2_manifest.json. No GPU outcomes yet. Scheduler query was sandbox-blocked; no CR10 fleet log yet. No submission/cancellation attempted.

## Second campaign: score preparation (2026-10-16 11:20 CDT)
Added scripts/dev2_analysis.py to enforce exact eight-prompt, seed-0 matching and finite required scores, then apply the frozen eligibility rule and two-variant cap. Full suite now 177 passed in 12.51s. Eight eval/IR task scripts copied and bash-checked in taskq/staging; each requires the actual 8 dev PNG/JSON pairs rather than stale pilot prerequisites in the source eval template. Generation tasks remain pending. No new GPU result.

## Second campaign: baseline validation and wait (2026-10-16 11:24 CDT)
Validated exactly the same eight dev IDs, seed 0, resolution 4096x4096. Released SEGA means: ImageReward 1.423017875, full-res MUSIQ 42.488059998, CLIP 30.320004225. Thus the CLIP eligibility floor is 30.020004225. At 11:23:53 all four generation tasks remain pending, none running/done/failed, and no CR10 log exists. This does not prove a scheduler state; direct scheduler access is sandbox-blocked. Scoring tasks remain staged until actual generation completes.

## Second campaign: independent review (2026-10-16 11:27 CDT)
Read-only review found no B1/B3 formula or composition bug. It identified that ImageReward CSVs omit seed; dev2_analysis.py now cross-checks exact eval/IR image paths and sidecar seed, prompt ID, resolution, method, steps, guidance, RoPE, extra-temperature setting, and B3 eta. New tests reject crossed paths and wrong protocol settings. Full suite: 183 passed in 14.31s. Generation implementation hashes remain unchanged from the frozen dev manifest.

## Conditional held-out labeling note (2026-10-16 11:49 CDT)
The evaluator keys identity by sidecar config.method, while ImageReward uses the output folder. If both sega_ctm eta values pass dev, merely separating their main2 folders would cause duplicate identities or mismatched merges. Add explicit variant labels or score separately and normalize identities before main2 analysis. No main2 work is authorized by the selection rule until actual dev scores pass.

## Second campaign: fleet wait (2026-10-16 13:35 CDT)
Four dev generation tasks have remained pending since 11:17 CDT, with no running/done/failed dev tasks. No CR10 fleet log; STOP is absent. Five-minute file checks continue. Direct scheduler access was sandbox-blocked; an optional status question at 12:01 remains unanswered. No new quality measurements, no selection decision, and no negative dev result. CPU implementation and 183 passing tests remain the completed work.

## Second campaign: immediate scoring handoff (2026-10-16 14:26 CDT)
Added success-only completion hooks to the four still-pending dev scripts. Each now atomically moves its prepared eval/IR scripts from staging to pending after all eight samples finish, reducing GPU idle time between stages. Claimed pending scripts atomically before editing; preserved original versions in taskq/hold/archived_20261016T142622_* and recorded hashes/paths in dev2_manifest.json. All four revised scripts passed bash -n. Generation commands, parameters, source code, and worker scripts are unchanged. No image or quality result yet.

## Second campaign: GPU generation started (2026-10-16 16:01 CDT)
All four dev tasks claimed at 15:58:26 CDT by externally launched fleet job 1076092, host c672-016 GPUs 0-3. Startup logs show pipeline/model loading with no errors. Telemetry is active at telemetry/c672-016_job1076092_20261016155826.csv. The originally recorded CR10 job 1075815 still has no visible log; current progress comes from 1076092. This session submitted/cancelled nothing. No completed image or quality metric verified yet.

## Second campaign: live GPU validation (2026-10-16 16:11 CDT)
Validated first five complete images per variant (20 total): frozen implementation/profile hashes, exact baseline prompt/seed/model/software fields, 4096 PNG sizes, 28 steps with exactly matching evaluated sigmas, finite paired-coordinate gains, alias time invariance/reference mean, B3 first-step parity/unit axis 0/active bounded correction. Receipt: dev2_generation_check.json. B3 maximum absolute logit corrections across these images: eta 0.5 = 0.244-0.470; eta 1.0 = 0.487-0.939. Generation 92-99 seconds/image on this worker, GPU utilization 92-100%.
Scheduler metadata caveat: pilot SEGA prompt 0 sidecar records max_image_seq_len=4096, while later records/new runs store 65536. generate() sets the value before executing the pipeline; all actual evaluated sigmas match exactly. Unordered private _use_default_values lists also differ in order. Original baseline records preserved. These are implementation/protocol checks, not efficacy results; await complete dev scoring.

## Second campaign: all dev images complete, scoring active (2026-10-16 16:22 CDT)
All four generation tasks succeeded with eight images each and immediately published their eight scoring tasks. Four evaluations are running; four IR tasks remain pending. Evaluation performs 195 reference-spectrum FFTs on CPU before any learned scorer, recomputing identical reference work in each new root. At 16:20 caches were advancing through 123-130 references; GPU memory/utilization were near zero during this CPU phase. Current running evaluator objects hold in-memory caches, so adding disk cache entries cannot accelerate them mid-run. For any future held-out evaluation, prepopulate its cache from a completed dev evaluation using unchanged reference/settings keys. No metric-based selection yet.

### 2026-10-16T16:32:20.507784-05:00: dev gate applied; held-out launched
All four dev roots have eight complete samples and successful eval/IR tasks. Scorer versions, provenance, revisions, and settings match pilot. Full dev selection table: `.agent/notes/dev2/dev_table.csv`. Only sega_ctm eta 1.0 passes (IR delta +0.025723625, full-res MUSIQ +0.017867804, CLIP +0.022446871). Eta 0.5 misses MUSIQ by 0.007850170; both alias variants lose IR. These scores are for selection only. Published 27 bash-checked 53_ scripts, 188 images for rows 7-194, with frozen code and eta 1.0. Final analysis excludes row 7. No new allocation, no commit.

### 2026-10-16T16:34:37.221226-05:00: held-out analysis and scoring prepared
Full suite: 194 passed in 16.61 s. main2/_v2 loading now verifies unique matched image paths, full 187-prompt coverage, finite scores, seed/resolution, and dev-row exclusion. Scoring tasks 62_eval_main2.sh and 62_ir_main2.sh are staged and bash-checked with all 376 image/sidecar prerequisites. Reused only reference data: 195 spectrum cache entries and 390 matching feature files, keyed by original reference file signatures and identical scorer settings. No generated-image scores copied. Receipt: main2_eval_preparation.json.

Read-only held-out review found no concrete issues. First four main2 samples passed frozen source/profile, baseline seed/model/software/sigma, PNG size, and bounded active-correction checks. Auditor: `.agent/notes/check_main2.py`; receipt: `main2_generation_check.json`. No inference implementation changed.

Held-out baseline preflight: 187 matched seed-0 SEGA samples, all ten metrics finite, evaluation/IR image paths agree. Full scorer versions, model revisions, metric provenance, crop settings, and device match the completed dev scoring. Receipt: `main2_baseline_check.json`.

Generation reached 56/188 images with eight completed chunks and no failures. To overlap CPU preprocessing with GPU generation, started an eight-thread, spectrum/tiling-only evaluation in separate `eval_krea/main2_cpu_prepass`. Its cache will be verified and merged before final GPU scoring. This changes scheduling only; metrics and image selection are unchanged.

CPU prepass 1 completed spectrum and tiling for 57 images. Exit 2 came only from W&B: sandbox sockets and the external W&B cache path are unavailable. Metric cache remains valid; preserve this failure in `main2_cpu_prepass_01.json`. Subsequent CPU-only prepasses use --no-wandb and local receipts. Final GPU scorer tasks retain normal W&B logging. No permission bypass or external path changes.

CPU prepass 2 completed 84 diagnostic records with zero errors using --no-wandb. This confirms the previous failure was logging only. Generation is at 96/188 images with no failed tasks.

At 122/188 generated images, queued bash-checked `62_00_gpu_prepass_main2.sh` after all 53_ tasks. It uses the otherwise spare GPU during the final three generation chunks to cache learned metrics and ImageReward on completed images. Separate prepass outputs are partial and will only seed the final all-image scoring pass. No model, prompt, metric, or selection rule changes. Final 62_eval_main2.sh and 62_ir_main2.sh remain staged with all-image prerequisites.

At 17:48 CDT, 174/188 images are complete with 24 successful generation chunks and no failures. Final three generation chunks occupy three GPUs; the fourth started learned-metric/IR preprocessing at 17:44:15, initially on 168 images. CPU diagnostics cover 164 images with zero errors. Protocol audit passed all 151 then-complete sidecars.

### 2026-10-16T17:59:41.861745-05:00: full held-out generation passed; final scoring published
All 27 chunks succeeded. All 188 PNG/JSON pairs passed source/profile hashes, prompt/seed/model/software/executed-sigma matching, PNG dimensions, finite paired gains, and bounded active correction. Final analysis excludes overlap row 7. Verified and merged CPU cache for 186 images, learned-score cache for 168 images, and 180 IR rows, preserving source paths/config hashes and scorer settings/revisions/provenance. Final 62_eval_main2.sh and 62_ir_main2.sh published at 17:58:57. Full receipts: main2_generation_check.json and main2_cache_merge.json. All local CPU sessions reaped; final quality remains unreported until complete scoring.

### 2026-10-16T18:08:51.678538-05:00: completed negative held-out result
Only sega_ctm eta 1.0 passed dev. All 188 images and scores validated; analysis excludes row 7 and uses 187 matched seed-0 prompts. Full suite: 194 passed. The candidate does not demonstrate an improvement over released SEGA. ImageReward/full-res MUSIQ CIs cross zero; PickScore and crop MUSIQ regress, and CLIP has a negative nominal CI. No equivalence or SOTA claim.

Dev values below are selection only (n=8):

| Variant | ImageReward | Full-res MUSIQ | CLIP | Advance |
|---|---:|---:|---:|---|
| sega_official | 1.423018 | 42.488060 | 30.320004 | Reference |
| alias_temp | 1.281098 | 35.807356 | 29.366986 | No |
| alias_temp_sref | 1.277426 | 44.972845 | 30.238460 | No |
| sega_ctm_eta05 | 1.441334 | 42.480210 | 30.200250 | No |
| sega_ctm_eta10 | 1.448742 | 42.505928 | 30.342451 | Yes |

Held-out paired result (n=187, 2,000 bootstrap draws, seed 0):

| Metric | SEGA mean | sega_ctm eta 1 mean | Paired delta [95% CI] |
|---|---:|---:|---:|
| ImageReward | 1.18994 | 1.19852 | +0.00857 [-0.00841, +0.02742] |
| PickScore | 23.05255 | 22.97729 | -0.07526 [-0.10234, -0.04914] |
| CLIP | 29.13552 | 29.04160 | -0.09392 [-0.19651, -0.00368] |
| MUSIQ, full resolution | 45.22784 | 45.01924 | -0.20860 [-0.70887, +0.31512] |
| MUSIQ, 1024 crops | 58.32778 | 57.19612 | -1.13166 [-1.38754, -0.90219] |
| CLIP-IQA, full resolution | 0.60299 | 0.60593 | +0.00294 [-0.00018, +0.00609] |
| CLIP-IQA, 1024 crops | 0.58597 | 0.58846 | +0.00250 [-0.00070, +0.00563] |
| Spectrum log RMSE | 1.06838 | 1.06596 | -0.00242 [-0.00905, +0.00433] |
| Spectrum slope error | 0.19995 | 0.19741 | -0.00254 [-0.00678, +0.00192] |
| Tiling at 1024 | 0.06836 | 0.06844 | +0.00008 [-0.00072, +0.00081] |

All generation and scoring tasks succeeded. STOP touched 18:05:35 CDT only after the entire pending/running queues were empty. Final GPU evaluation logged offline W&B n8xhe6iy. Critical score tables, raw reports, and prompt manifest archived in `.agent/notes/main2`; full report `main2/REPORT.md`. No commit because .git is read-only. No new experiments queued.

Final closure: independent read-only review reproduced every one of the ten held-out paired mean deltas and 2,000-draw seed-0 percentile CIs from raw CSVs within 1e-10. Exactly 187 matched seed-0 prompts 8-194; row 7 excluded; no discrepancies. All four fleet workers exited between 18:05:41 and 18:06:06 CDT after STOP. Entire pending/running queues empty, local CPU sessions reaped, git diff --check passed. Slurm allocation status was not queried. Campaign complete; no further experiments queued.
