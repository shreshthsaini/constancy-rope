# Strengthening campaign results

Written by the Claude analyst. The summary above the marker is hand-written from the generated tables below it;
every number in it is copied from those tables or from the analysis CSVs named there. Regenerate the tables with
`crope_py scripts/strengthen_report.py` (it keeps this header).

## Summary (hand-written)

Status 2026-10-24 (final). All six Krea blocks were scored (22 of 22 evaluation tasks done, 0 failed), merged and
analysed by `scripts/strengthen_confirm.py run`. Every number below is copied from the tables under the marker or
from `eval_krea/strengthen/analysis_*.csv`. An independent recomputation from the raw evaluator CSVs
(`verification/independent_check.py`, no project imports) reproduces all 223 contrast cells and the registered
family within 1e-14; 71 paper macros match the CSVs (`verification/crosscheck.json`). Evaluator versions, model
revisions and metric schema are identical to the original campaign (eval_krea/main).

Registered outcomes (amended preregistration; Holm over H1_IR, H1_Pick, H2):
- H1, `temp_only_sbase` equivalent to `sega_official`, pooled 187 held-out + 150 fresh prompts (n=337): SUPPORTED.
  ImageReward +0.0095 [-0.0201, +0.0408] inside +/-0.0820, Holm p 0.0015; PickScore -0.0167 [-0.0595, +0.0256]
  inside +/-0.10, Holm p 0.0015. The pool reuses 187 inspected prompts: an extension, not independent confirmation.
- Fresh 150 alone (descriptive, same rules): ImageReward -0.0112 [-0.0554, +0.0346] inside its margin; PickScore
  -0.0725 [-0.1331, -0.0106] crosses -0.10, so equivalence is NOT established on fresh prompts alone.
- H2, `sega_official` minus `temp_only_sbase` full-frame MUSIQ > 0 on fresh 150: SUPPORTED, +1.558 [+0.426, +2.718],
  Holm p 0.0045.
- H3 withdrawn before generation; historical negative results stand.

Factorial (P1, held-out rows 8-194, seed 0, Holm family 80; 31 survive):
- Temperature main effect carries the gain: ImageReward +0.594 [+0.506, +0.686], PickScore +1.036 [+0.920, +1.157],
  full MUSIQ +13.45 [+11.96, +15.11], crop MUSIQ +18.07 [+16.35, +19.83]; 27 Holm hits (all metrics except tiling,
  in the main effect and both simple effects). Temperature on NTK base +0.603, on SEGA base +0.585 (ImageReward).
- Base main effect: ImageReward +0.027 [-0.003, +0.056], full MUSIQ +0.20 [-0.29, +0.72]; crop MUSIQ +0.52
  [+0.16, +0.86] nominal only; 0 Holm hits for the base (main or simple). Interaction: 0 Holm hits (CLIP +0.38
  [+0.12, +0.66] nominal).
- Allocation (released SEGA minus SEGA base + temp.): full MUSIQ +2.09 [+0.97, +3.25], crop CLIP-IQA +0.017, slope
  error -0.024 (better), tiling +0.0068 (worse), all Holm; ImageReward -0.026 [-0.068, +0.014], PickScore -0.03
  [-0.09, +0.03] tie.
- Identity: base + temperature + allocation = SEGA minus NTK (ImageReward +0.594, full MUSIQ +15.74).

Temperature sweep (P3, rows 8-62, n=55, Holm 40; 21 survive): ImageReward 0.476 (1.0), 1.031 (1.1), 1.148
(s^0.16), 1.179 (4/3), 1.017 (1.45). No multiplier beats s^0.16 under Holm on ImageReward, PickScore or full MUSIQ.
Full MUSIQ peaks at s^0.16 (41.97; 30.90 at 1.45); crop MUSIQ rises monotonically, 40.12 to 65.87.

Fresh arms (E1, Holm 50; 20 survive): DyPE beats SEGA (released) under Holm on crop MUSIQ +2.32 [+1.41, +3.25] and
CLIP-IQA +0.024 [+0.011, +0.036] and loses ImageReward -0.166 and PickScore -0.38. NTK + temp. ties ImageReward
(-0.042 [-0.093, +0.008]) and loses PickScore -0.12 [-0.18, -0.05] (Holm). SEGA base + temp.: ImageReward -0.011
tie, PickScore -0.07 [-0.13, -0.01] nominal, crop CLIP-IQA and slope error Holm-worse. No arm beats SEGA (released)
on any reward model.

Seeds (E2, 187 prompts, seeds averaged within prompt, Holm 10): SEGA base + temp. minus SEGA (released) ImageReward
+0.021 [-0.004, +0.044], PickScore -0.000 [-0.035, +0.037], full MUSIQ -2.00 [-2.74, -1.30] (Holm). Seeds 1-2:
ImageReward +0.018 [-0.007, +0.042], full MUSIQ -1.96 [-2.81, -1.20]. Each seed alone: MUSIQ interval below zero.

Checkpoint x guidance (P2, 8 dev prompts, official DyPE, seed 42; descriptive): FLUX.1-dev / Krea lattice power
12.0x [7.8, 18.6] at guidance 3.5, 7.4x [4.5, 12.7] at 4.5; shared bin, guidance 4.5/3.5: 0.52x [0.41, 0.64] on
FLUX.1-dev, 0.89x [0.67, 1.11] on Krea. FLUX.1-dev minus Krea ImageReward -2.078 (g3.5), -1.584 (g4.5).

Default flow shift (P7, 8 dev prompts, seed 0; descriptive): mu 11.55 leaves the last model call at sigma 0.9997
(official 0.105, 3 calls below 0.3). ImageReward -3.701 [-3.959, -3.419], full MUSIQ -20.05 [-24.68, -13.86]; all
ten metrics worse.

Qwen-Image (E3): not run. 0 of 450 images; all 60 tasks failed with CUDA OOM next to a co-resident ~45 GiB process
(`qwen_failure_diagnosis.json`). No transfer claim.

"Beats SEGA" audit: the only Holm-surviving wins over SEGA (released) are DyPE on crop MUSIQ and CLIP-IQA (fresh,
family 50), with Holm losses on ImageReward and PickScore. Nothing beats SEGA (released) on a reward model.

<!-- AUTO-TABLES: everything below is regenerated by scripts/strengthen_report.py -->

# Strengthening results (generated by scripts/strengthen_report.py)

Every number below is copied from the analysis directories under .agent/notes/strengthen/analysis/. Delta = arm minus reference (released SEGA unless stated). tie = nominal 95% CI contains 0. H = Holm-adjusted p <= 0.05 within the registered family.

Block status: P1=complete, P3=complete, E1=complete, E2=complete, E3=missing, ckpt_matrix=complete, default_shift=complete

## (a) Factorial base x temperature plus allocation (P1, held-out rows 8-194, seed 0)
Complete block: True. Holm family 80 (8 contrasts x 10 metrics), two-sided absolute-tail bootstrap p. Cells: delta [95% CI] verdict; H = survives Holm.
| Contrast | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_main_effect | +0.027 [-0.003, +0.056] tie | +0.017 [-0.026, +0.059] tie | -0.021 [-0.149, +0.107] tie | +0.20 [-0.29, +0.72] tie | +0.52 [+0.16, +0.86] better | +0.0010 [-0.0023, +0.0045] tie | +0.0006 [-0.0033, +0.0044] tie | -0.0048 [-0.0190, +0.0068] tie | -0.0002 [-0.0035, +0.0033] tie | -0.0039 [-0.0066, -0.0012] better |
| temperature_main_effect | +0.594 [+0.506, +0.686] better H | +1.036 [+0.920, +1.157] better H | +2.028 [+1.614, +2.469] better H | +13.45 [+11.96, +15.11] better H | +18.07 [+16.35, +19.83] better H | +0.0355 [+0.0205, +0.0503] better H | +0.0317 [+0.0162, +0.0472] better H | -0.1778 [-0.2518, -0.1110] better H | +0.0586 [+0.0389, +0.0790] worse H | -0.0102 [-0.0228, +0.0020] tie |
| interaction | -0.018 [-0.087, +0.047] tie | +0.005 [-0.071, +0.082] tie | +0.382 [+0.117, +0.657] better | -0.13 [-1.23, +0.94] tie | -0.38 [-1.13, +0.27] tie | -0.0015 [-0.0071, +0.0046] tie | -0.0000 [-0.0070, +0.0067] tie | +0.0006 [-0.0230, +0.0268] tie | -0.0077 [-0.0141, -0.0016] better | +0.0013 [-0.0030, +0.0056] tie |
| allocation | -0.026 [-0.068, +0.014] tie | -0.028 [-0.089, +0.029] tie | -0.019 [-0.228, +0.188] tie | +2.09 [+0.97, +3.25] better H | +0.11 [-0.48, +0.71] tie | +0.0093 [+0.0011, +0.0178] better | +0.0167 [+0.0078, +0.0255] better H | -0.0338 [-0.0520, -0.0155] better | -0.0237 [-0.0343, -0.0132] better H | +0.0068 [+0.0028, +0.0106] worse H |
| base_at_none | +0.036 [-0.015, +0.088] tie | +0.014 [-0.046, +0.074] tie | -0.212 [-0.418, -0.015] worse | +0.27 [-0.23, +0.79] tie | +0.71 [+0.14, +1.38] better | +0.0018 [-0.0013, +0.0050] tie | +0.0006 [-0.0029, +0.0040] tie | -0.0051 [-0.0306, +0.0159] tie | +0.0037 [-0.0001, +0.0076] tie | -0.0046 [-0.0075, -0.0016] better |
| base_at_sref | +0.018 [-0.021, +0.057] tie | +0.019 [-0.033, +0.071] tie | +0.170 [+0.000, +0.341] better | +0.13 [-0.77, +1.01] tie | +0.33 [-0.03, +0.68] tie | +0.0003 [-0.0049, +0.0059] tie | +0.0006 [-0.0059, +0.0068] tie | -0.0045 [-0.0152, +0.0059] tie | -0.0040 [-0.0094, +0.0015] tie | -0.0032 [-0.0072, +0.0007] tie |
| temperature_at_ntk2 | +0.603 [+0.508, +0.702] better H | +1.034 [+0.913, +1.159] better H | +1.837 [+1.394, +2.294] better H | +13.52 [+11.93, +15.27] better H | +18.26 [+16.46, +20.09] better H | +0.0363 [+0.0213, +0.0509] better H | +0.0317 [+0.0158, +0.0469] better H | -0.1781 [-0.2538, -0.1078] better H | +0.0624 [+0.0430, +0.0831] worse H | -0.0109 [-0.0238, +0.0018] tie |
| temperature_at_sega | +0.585 [+0.495, +0.679] better H | +1.039 [+0.918, +1.165] better H | +2.219 [+1.771, +2.674] better H | +13.39 [+11.76, +15.06] better H | +17.88 [+16.17, +19.63] better H | +0.0348 [+0.0194, +0.0501] better H | +0.0317 [+0.0161, +0.0483] better H | -0.1775 [-0.2496, -0.1118] better H | +0.0547 [+0.0343, +0.0754] worse H | -0.0096 [-0.0222, +0.0029] tie |

Arm means (95% CI omitted here; see CSV):
| Method | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ntk2 | 0.596 (n=187) | 22.028 (n=187) | 27.147 (n=187) | 29.49 (n=187) | 39.63 (n=187) | 0.5571 (n=187) | 0.5370 (n=187) | 1.2848 (n=187) | 0.1653 (n=187) | 0.0757 (n=187) |
| ntk2_sref | 1.198 (n=187) | 23.061 (n=187) | 28.984 (n=187) | 43.01 (n=187) | 57.89 (n=187) | 0.5934 (n=187) | 0.5687 (n=187) | 1.1067 (n=187) | 0.2277 (n=187) | 0.0648 (n=187) |
| sega_base_none | 0.631 (n=187) | 22.042 (n=187) | 26.935 (n=187) | 29.76 (n=187) | 40.34 (n=187) | 0.5589 (n=187) | 0.5376 (n=187) | 1.2796 (n=187) | 0.1690 (n=187) | 0.0712 (n=187) |
| temp_only_sbase | 1.216 (n=187) | 23.081 (n=187) | 29.154 (n=187) | 43.14 (n=187) | 58.22 (n=187) | 0.5937 (n=187) | 0.5693 (n=187) | 1.1022 (n=187) | 0.2237 (n=187) | 0.0616 (n=187) |
| sega_official | 1.190 (n=187) | 23.053 (n=187) | 29.136 (n=187) | 45.23 (n=187) | 58.33 (n=187) | 0.6030 (n=187) | 0.5860 (n=187) | 1.0684 (n=187) | 0.1999 (n=187) | 0.0684 (n=187) |

## (b) Checkpoint x guidance matrix (P2, official DyPE, dev rows 0-7, seed 42)
Lattice statistic: FLUX.1-dev / Krea whitened power at the dev-selected peak (1/64 < f < 1/8), geometric mean over 8 prompts, bootstrap CI on logs:
- guidance 3.5: ratio 12.00 [7.82, 18.58] (n=8)
- guidance 4.5: ratio 7.37 [4.50, 12.68] (n=8)
Paired contrasts (descriptive, no multiplicity correction):
- dev_minus_krea_g3p5 IR: -2.078 [-2.616, -1.452] worse
- dev_minus_krea_g3p5 Pick: -3.029 [-3.311, -2.664] worse
- dev_minus_krea_g3p5 CLIP: -5.532 [-8.582, -2.883] worse
- dev_minus_krea_g3p5 MUSIQ: -7.02 [-15.68, +1.78] tie
- dev_minus_krea_g3p5 MUSIQ crop: -12.01 [-17.93, -6.47] worse
- dev_minus_krea_g4p5 IR: -1.584 [-2.177, -0.988] worse
- dev_minus_krea_g4p5 Pick: -2.412 [-3.006, -1.897] worse
- dev_minus_krea_g4p5 CLIP: -3.201 [-5.141, -1.548] worse
- dev_minus_krea_g4p5 MUSIQ: -8.25 [-14.34, -2.34] worse
- dev_minus_krea_g4p5 MUSIQ crop: -13.53 [-20.18, -6.90] worse
- g4p5_minus_g3p5_dev IR: +0.543 [+0.035, +1.097] better
- g4p5_minus_g3p5_dev Pick: +0.502 [+0.091, +0.926] better
- g4p5_minus_g3p5_dev CLIP: +2.019 [-0.111, +4.795] tie
- g4p5_minus_g3p5_dev MUSIQ: -0.46 [-2.72, +1.72] tie
- g4p5_minus_g3p5_dev MUSIQ crop: -1.72 [-5.63, +2.35] tie
- g4p5_minus_g3p5_krea IR: +0.049 [-0.041, +0.147] tie
- g4p5_minus_g3p5_krea Pick: -0.115 [-0.411, +0.125] tie
- g4p5_minus_g3p5_krea CLIP: -0.311 [-0.942, +0.239] tie
- g4p5_minus_g3p5_krea MUSIQ: +0.76 [-3.20, +5.29] tie
- g4p5_minus_g3p5_krea MUSIQ crop: -0.19 [-1.57, +1.25] tie
- guidance_effect_dev_minus_krea IR: +0.494 [+0.068, +0.957] better
- guidance_effect_dev_minus_krea Pick: +0.617 [+0.190, +0.985] better
- guidance_effect_dev_minus_krea CLIP: +2.330 [+0.238, +5.106] better
- guidance_effect_dev_minus_krea MUSIQ: -1.22 [-7.28, +4.00] tie
- guidance_effect_dev_minus_krea MUSIQ crop: -1.53 [-5.35, +2.17] tie

## (c) Temperature sweep (P3, SEGA base, temp_only, held-out rows 8-62, seed 0)
Complete block: True. Holm family 40 (4 settings minus sref x 10 metrics).
| Method | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tau1p0 | 0.476 (n=55) | 21.745 (n=55) | 26.779 (n=55) | 30.53 (n=55) | 40.12 (n=55) | 0.5512 (n=55) | 0.5296 (n=55) | 1.2323 (n=55) | 0.1677 (n=55) | 0.0726 (n=55) |
| tau1p1 | 1.031 (n=55) | 22.491 (n=55) | 28.829 (n=55) | 40.49 (n=55) | 49.22 (n=55) | 0.5880 (n=55) | 0.5713 (n=55) | 1.0086 (n=55) | 0.1939 (n=55) | 0.0634 (n=55) |
| tau_sref | 1.148 (n=55) | 22.884 (n=55) | 29.336 (n=55) | 41.97 (n=55) | 59.68 (n=55) | 0.5946 (n=55) | 0.5797 (n=55) | 1.1653 (n=55) | 0.2339 (n=55) | 0.0596 (n=55) |
| tau_logn | 1.179 (n=55) | 22.788 (n=55) | 29.542 (n=55) | 39.88 (n=55) | 63.09 (n=55) | 0.6090 (n=55) | 0.5885 (n=55) | 1.3322 (n=55) | 0.2696 (n=55) | 0.0548 (n=55) |
| tau1p45 | 1.017 (n=55) | 22.242 (n=55) | 28.539 (n=55) | 30.90 (n=55) | 65.87 (n=55) | 0.6664 (n=55) | 0.6382 (n=55) | 1.7302 (n=55) | 0.4121 (n=55) | 0.0374 (n=55) |

Each setting minus sref (multiplier 4**0.16 = 1.2483):
| Contrast | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tau1p0 | -0.671 [-0.883, -0.463] worse H | -1.139 [-1.390, -0.893] worse H | -2.557 [-3.535, -1.588] worse H | -11.43 [-14.68, -8.30] worse H | -19.56 [-22.79, -16.16] worse H | -0.0433 [-0.0726, -0.0151] worse H | -0.0501 [-0.0786, -0.0208] worse H | +0.0670 [-0.0706, +0.2034] tie | -0.0662 [-0.1043, -0.0299] better H | +0.0130 [-0.0082, +0.0332] tie |
| tau1p1 | -0.117 [-0.246, +0.032] tie | -0.393 [-0.550, -0.231] worse H | -0.507 [-1.058, +0.048] tie | -1.48 [-4.73, +1.64] tie | -10.46 [-12.41, -8.66] worse H | -0.0066 [-0.0343, +0.0213] tie | -0.0084 [-0.0355, +0.0202] tie | -0.1567 [-0.2446, -0.0712] better H | -0.0400 [-0.0694, -0.0116] better | +0.0038 [-0.0088, +0.0147] tie |
| tau_logn | +0.032 [-0.067, +0.140] tie | -0.096 [-0.238, +0.067] tie | +0.206 [-0.247, +0.673] tie | -2.08 [-5.32, +1.15] tie | +3.41 [+2.18, +4.70] better H | +0.0144 [-0.0041, +0.0332] tie | +0.0088 [-0.0095, +0.0278] tie | +0.1669 [+0.1132, +0.2187] worse H | +0.0357 [+0.0087, +0.0621] worse | -0.0048 [-0.0120, +0.0034] tie |
| tau1p45 | -0.131 [-0.260, +0.009] tie | -0.642 [-0.836, -0.439] worse H | -0.797 [-1.420, -0.148] worse | -11.06 [-14.39, -7.53] worse H | +6.19 [+4.10, +8.35] better H | +0.0718 [+0.0453, +0.0968] better H | +0.0586 [+0.0296, +0.0832] better H | +0.5649 [+0.4746, +0.6499] worse H | +0.1782 [+0.1348, +0.2183] worse H | -0.0222 [-0.0328, -0.0108] better H |

## (d) Fresh150 confirmation (E1) and amended H1/H2
Complete block: True.
Amended confirmatory family (Holm 3): H1 pooled historical 187 + fresh 150; H2 fresh 150. Overall H1 **supported**, H2 **supported**, H3 not_tested_historical_only.
- H1_IR (n=337, margin +/-0.0820): delta +0.009 [-0.020, +0.041], p 0.0004998, Holm p 0.001499: **supported**
- H1_Pick (n=337, margin +/-0.1000): delta -0.017 [-0.059, +0.026], p 0.0004998, Holm p 0.001499: **supported**
- H2_MUSIQ (n=150): delta +1.56 [+0.43, +2.72], p 0.004498, Holm p 0.004498: **supported**
Fresh-only descriptive summary (not a substitute for the pooled test): H1 not_established, H2 supported.
- H1_IR (n=150, margin +/-0.0820): delta -0.011 [-0.055, +0.035], p 0.0009995, Holm p 0.002999: **supported**
- H1_Pick (n=150, margin +/-0.1000): delta -0.073 [-0.133, -0.011], p 0.1994, Holm p 0.1994: **not_established**
- H2_MUSIQ (n=150): delta +1.56 [+0.43, +2.72], p 0.004498, Holm p 0.008996: **supported**

Secondary: each arm minus released SEGA, Holm family 50:
| Contrast | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temp_only_sbase | -0.011 [-0.055, +0.035] tie | -0.073 [-0.133, -0.011] worse | -0.182 [-0.453, +0.093] tie | -1.56 [-2.72, -0.43] worse | -0.64 [-1.32, +0.03] tie | -0.0074 [-0.0173, +0.0020] tie | -0.0231 [-0.0332, -0.0133] worse H | +0.0050 [-0.0346, +0.0381] tie | +0.0161 [+0.0063, +0.0256] worse H | -0.0058 [-0.0094, -0.0022] better |
| dype | -0.166 [-0.250, -0.084] worse H | -0.378 [-0.513, -0.258] worse H | -0.116 [-0.476, +0.246] tie | -0.65 [-1.99, +0.74] tie | +2.32 [+1.41, +3.25] better H | +0.0235 [+0.0115, +0.0359] better H | +0.0115 [-0.0022, +0.0252] tie | -0.0420 [-0.1012, +0.0112] tie | +0.0160 [+0.0003, +0.0314] worse | +0.0090 [-0.0026, +0.0214] tie |
| ntk2 | -0.923 [-1.077, -0.768] worse H | -1.053 [-1.216, -0.891] worse H | -2.515 [-3.054, -1.981] worse H | -10.48 [-12.46, -8.53] worse H | -13.00 [-14.75, -11.29] worse H | -0.0206 [-0.0376, -0.0034] worse | -0.0180 [-0.0351, -0.0017] worse | +0.4342 [+0.2943, +0.5825] worse H | +0.0083 [-0.0199, +0.0384] tie | -0.0062 [-0.0175, +0.0048] tie |
| sega_base_none | -0.875 [-1.022, -0.723] worse H | -1.054 [-1.217, -0.894] worse H | -2.501 [-3.047, -1.966] worse H | -10.28 [-12.15, -8.33] worse H | -13.15 [-14.99, -11.42] worse H | -0.0243 [-0.0405, -0.0066] worse | -0.0237 [-0.0403, -0.0079] worse | +0.4694 [+0.3441, +0.6082] worse H | +0.0128 [-0.0161, +0.0425] tie | -0.0107 [-0.0225, +0.0004] tie |
| ntk2_sref | -0.042 [-0.093, +0.008] tie | -0.117 [-0.185, -0.055] worse H | -0.252 [-0.522, +0.002] tie | -0.76 [-1.99, +0.49] tie | -0.55 [-1.15, +0.07] tie | -0.0118 [-0.0219, -0.0017] worse | -0.0277 [-0.0392, -0.0167] worse H | +0.0135 [-0.0278, +0.0502] tie | +0.0117 [-0.0031, +0.0258] tie | -0.0014 [-0.0057, +0.0030] tie |

## (e) Seeds 0/1/2 (E2, held-out rows 8-194)
Complete block: True. Primary: prompt-level mean over seeds 0,1,2, then paired bootstrap (n=187 prompts). Holm family 10 (temp_only_sbase minus SEGA x 10 metrics).

seeds012:
| Contrast | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temp_only_sbase | +0.021 [-0.004, +0.044] tie | -0.000 [-0.035, +0.037] tie | -0.021 [-0.179, +0.124] tie | -2.00 [-2.74, -1.30] worse H | -0.19 [-0.69, +0.30] tie | -0.0066 [-0.0131, -0.0005] worse | -0.0142 [-0.0209, -0.0074] worse H | +0.0449 [+0.0293, +0.0607] worse H | +0.0235 [+0.0143, +0.0329] worse H | -0.0060 [-0.0085, -0.0033] better H |

seeds12:
| Contrast | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temp_only_sbase | +0.018 [-0.007, +0.042] tie | -0.014 [-0.057, +0.027] tie | -0.041 [-0.222, +0.129] tie | -1.96 [-2.81, -1.20] worse H | -0.23 [-0.79, +0.35] tie | -0.0053 [-0.0123, +0.0018] tie | -0.0130 [-0.0204, -0.0053] worse H | +0.0505 [+0.0333, +0.0677] worse H | +0.0233 [+0.0136, +0.0335] worse H | -0.0056 [-0.0088, -0.0024] better H |

seed0:
| Contrast | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temp_only_sbase | +0.026 [-0.014, +0.068] tie | +0.028 [-0.029, +0.089] tie | +0.019 [-0.188, +0.228] tie | -2.09 [-3.25, -0.97] worse H | -0.11 [-0.71, +0.48] tie | -0.0093 [-0.0178, -0.0011] worse | -0.0167 [-0.0255, -0.0078] worse H | +0.0338 [+0.0155, +0.0520] worse H | +0.0237 [+0.0132, +0.0343] worse H | -0.0068 [-0.0106, -0.0028] better H |

seed1:
| Contrast | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temp_only_sbase | +0.008 [-0.023, +0.038] tie | +0.009 [-0.052, +0.069] tie | -0.141 [-0.376, +0.083] tie | -2.14 [-3.23, -1.14] worse H | -0.22 [-0.89, +0.46] tie | -0.0067 [-0.0149, +0.0015] tie | -0.0141 [-0.0234, -0.0053] worse H | +0.0545 [+0.0346, +0.0736] worse H | +0.0228 [+0.0114, +0.0338] worse H | -0.0033 [-0.0076, +0.0011] tie |

seed2:
| Contrast | IR | Pick | CLIP | MUSIQ | MUSIQ crop | CLIP-IQA | CLIP-IQA crop | Spec. RMSE | Slope err. | Tiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temp_only_sbase | +0.029 [-0.005, +0.064] tie | -0.038 [-0.098, +0.019] tie | +0.060 [-0.147, +0.270] tie | -1.77 [-2.66, -0.98] worse H | -0.23 [-0.86, +0.46] tie | -0.0039 [-0.0124, +0.0043] tie | -0.0119 [-0.0203, -0.0032] worse H | +0.0465 [+0.0261, +0.0665] worse H | +0.0238 [+0.0133, +0.0350] worse H | -0.0078 [-0.0123, -0.0040] better H |
Replication summary seeds012: H1 supported, H2 supported.
- H1_IR (n=187, margin +/-0.0820): delta +0.021 [-0.004, +0.044], p 0.0004998, Holm p 0.001499: **supported**
- H1_Pick (n=187, margin +/-0.1000): delta -0.000 [-0.035, +0.037], p 0.0004998, Holm p 0.001499: **supported**
- H2_MUSIQ (n=187): delta +2.00 [+1.30, +2.74], p 0.0004998, Holm p 0.001499: **supported**
Replication summary seeds12: H1 supported, H2 supported.
- H1_IR (n=187, margin +/-0.0820): delta +0.018 [-0.007, +0.042], p 0.0004998, Holm p 0.001499: **supported**
- H1_Pick (n=187, margin +/-0.1000): delta -0.014 [-0.057, +0.027], p 0.0004998, Holm p 0.001499: **supported**
- H2_MUSIQ (n=187): delta +1.96 [+1.20, +2.81], p 0.0004998, Holm p 0.001499: **supported**

## (f) Qwen-Image transfer (E3, fresh150, seed 0, exploratory)
Not analysed yet.

## P7 default flow shift (dev rows 0-7, seed 0, descriptive)
Paired contrasts (descriptive, no multiplicity correction):
- default_minus_official_shift IR: -3.701 [-3.959, -3.419] worse
- default_minus_official_shift Pick: -6.221 [-7.108, -5.422] worse
- default_minus_official_shift CLIP: -15.702 [-18.549, -12.646] worse
- default_minus_official_shift MUSIQ: -20.05 [-24.68, -13.86] worse
- default_minus_official_shift MUSIQ crop: -24.42 [-29.71, -19.39] worse

