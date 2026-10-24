# Strengthening campaign: numbers digest (2026-10-24, final)

One line per finding, ready for the abstract and introduction. Intervals are nominal 95% paired prompt-bootstrap
intervals (2,000 draws, seed 0); "Holm" means the Holm-adjusted p <= 0.05 inside the block's registered family.
Macro names (tables/num_strengthen.tex) follow each line; quote the macro, not the typed number. Source CSVs:
/scratch/09032/saini_2/constancy_rope/eval_krea/strengthen/analysis_*.csv. Verified: verification/.

## Registered confirmation (preregistered, Holm over three tests)

1. H1 supported: on 337 pooled prompts, SEGA base + temp. matches released SEGA within the registered margins: ImageReward +0.009 [-0.020, +0.041] (margin +/-0.082, Holm p 0.0015) and PickScore -0.017 [-0.059, +0.026] (margin +/-0.10, Holm p 0.0015). `st@h@d@ir` `st@h@ci@ir` `st@h@p@ir` `st@h@d@pick` `st@h@ci@pick` `st@h@p@pick` `st@h@H1`
2. H1 caveat: the pool reuses the 187 held-out prompts already inspected; on the 150 fresh prompts alone PickScore is -0.073 [-0.133, -0.011], which crosses the -0.10 margin, so fresh-only equivalence is not established (ImageReward -0.011 [-0.055, +0.035] stays inside). `st@hf@d@pick` `st@hf@ci@pick` `st@hf@status@pick` `st@hf@d@ir` `st@hf@ci@ir`
3. H2 supported: on 150 fresh prompts the per-band allocation adds +1.56 [+0.43, +2.72] full-frame MUSIQ (Holm p 0.0045). `st@h@d@musiq` `st@h@ci@musiq` `st@h@p@musiq` `st@h@H2`
4. H3 was withdrawn before generation; its two controllers had already lost held out.

## Factorial decomposition (held-out rows 8-194, n=187, seed 0, Holm over 80 tests; 31 survive)

5. The global temperature carries released SEGA's gain over NTK: +0.594 [+0.506, +0.686] ImageReward of a +0.594 total, and +13.45 [+11.96, +15.11] of +15.74 full-frame MUSIQ (Holm). `st@f@d@temp@ir` `st@f@ci@temp@ir` `st@f@tot@ir` `st@f@d@temp@musiq` `st@f@ci@temp@musiq` `st@f@tot@musiq`
6. The temperature also raises PickScore by +1.04 [+0.92, +1.16] and crop MUSIQ by +18.07 [+16.35, +19.83] (Holm). `st@f@d@temp@pick` `st@f@ci@temp@pick` `st@f@d@temp@musiqc` `st@f@ci@temp@musiqc`
7. SEGA's RoPE base adds nothing detectable: ImageReward +0.027 [-0.003, +0.056], full-frame MUSIQ +0.20 [-0.29, +0.72]; no base test survives Holm on any of ten metrics (crop MUSIQ +0.52 [+0.16, +0.86] is nominal only). `st@f@d@base@ir` `st@f@ci@base@ir` `st@f@d@base@musiq` `st@f@ci@base@musiq` `st@f@d@base@musiqc` `st@f@nholm@base`
8. Base and temperature do not interact: no interaction test survives Holm; the temperature raises ImageReward by +0.603 on the NTK base and +0.585 on the SEGA base. `st@f@nholm@inter` `st@f@d@tempntk@ir` `st@f@d@tempsega@ir`
9. The allocation adds +2.09 [+0.97, +3.25] full-frame MUSIQ (Holm) and ties ImageReward (-0.026 [-0.068, +0.014]) and PickScore (-0.03 [-0.09, +0.03]). `st@f@d@alloc@musiq` `st@f@ci@alloc@musiq` `st@f@d@alloc@ir` `st@f@ci@alloc@ir` `st@f@d@alloc@pick` `st@f@ci@alloc@pick`
10. Its other Holm effects: crop CLIP-IQA +0.017, spectral slope error -0.024 (better), tiling +0.0068 (worse). `st@f@d@alloc@iqac` `st@f@d@alloc@slope` `st@f@d@alloc@tile`

## Temperature sweep (SEGA base, no allocation, rows 8-62, n=55, Holm over 40; 21 survive)

11. ImageReward is 0.476 without temperature, 1.031 at 1.1, 1.148 at s^0.16, 1.179 at 4/3 and 1.017 at 1.45; no multiplier beats s^0.16 under Holm on ImageReward, PickScore or full-frame MUSIQ. `st@sm@t100@ir` `st@sm@t110@ir` `st@sm@tsref@ir` `st@sm@tlogn@ir` `st@sm@t145@ir`
12. Full-frame MUSIQ peaks at s^0.16 (41.97; 30.90 at 1.45) while crop MUSIQ rises at every step (40.12 to 65.87), so crop scores favour hotter attention than full-frame scores and reward models. `st@sm@tsref@musiq` `st@sm@t145@musiq` `st@sm@t100@musiqc` `st@sm@t145@musiqc`

## Fresh prompts (150 unseen Aesthetic-4K captions, seed 0, Holm over 50; 20 survive)

13. Released SEGA keeps its held-out lead over the baselines: ImageReward differences -0.923 (NTK), -0.875 (SEGA base alone), -0.166 (DyPE), all Holm. `st@e@d@ntk@ir` `st@e@d@segabase@ir` `st@e@d@dype@ir`
14. The only Holm wins over released SEGA belong to DyPE, on crop MUSIQ +2.32 [+1.41, +3.25] and CLIP-IQA +0.024 [+0.011, +0.036], and DyPE loses ImageReward (-0.166) and PickScore (-0.38). Nothing beats released SEGA on a reward model. `st@e@d@dype@musiqc` `st@e@ci@dype@musiqc` `st@e@d@dype@iqa` `st@e@ci@dype@iqa` `st@e@d@dype@pick`
15. Both temperature arms trail released SEGA on fresh PickScore: -0.07 [-0.13, -0.01] on the SEGA base (nominal) and -0.12 [-0.18, -0.05] on the NTK base (Holm); NTK + temp. ties ImageReward (-0.042 [-0.093, +0.008]). `st@e@d@tsb@pick` `st@e@ci@tsb@pick` `st@e@d@ntktemp@pick` `st@e@ci@ntktemp@pick` `st@e@d@ntktemp@ir` `st@e@ci@ntktemp@ir`

## Seed replication (held-out, 187 prompts, seeds 0-2 averaged within prompt, Holm over 10)

16. SEGA base + temp. minus released SEGA: ImageReward +0.021 [-0.004, +0.044], PickScore -0.000 [-0.035, +0.037], full-frame MUSIQ -2.00 [-2.74, -1.30] (Holm). `st@rseeds012@d@tsb@ir` `st@rseeds012@ci@tsb@ir` `st@hr@d@pick` `st@hr@ci@pick` `st@rseeds012@d@tsb@musiq` `st@rseeds012@ci@tsb@musiq`
17. Seeds 1 and 2 alone: ImageReward +0.018 [-0.007, +0.042], full-frame MUSIQ -1.96 [-2.81, -1.20]; every single seed puts the MUSIQ interval below zero. `st@rseeds12@d@tsb@ir` `st@rseeds12@ci@tsb@ir` `st@rseeds12@d@tsb@musiq` `st@rseeds12@ci@tsb@musiq`

## Controls (development prompts 0-7, descriptive)

18. The dot lattice is a checkpoint effect: at matched guidance FLUX.1-dev carries 12.0x [7.8, 18.6] the Krea lattice power at guidance 3.5 and 7.4x [4.5, 12.7] at 4.5 (official DyPE code, seed 42). `st@k@ratio@g3p5` `st@k@ci@g3p5` `st@k@ratio@g4p5` `st@k@ci@g4p5`
19. Guidance 4.5 versus 3.5 scales the FLUX.1-dev lattice by 0.52x [0.41, 0.64] and the Krea lattice by 0.89x [0.67, 1.11] (shared bin). `st@k@two@gdev` `st@k@twoci@gdev` `st@k@two@gkrea` `st@k@twoci@gkrea`
20. The diffusers default flow shift (mu 11.55, last sigma 0.9997) lowers released SEGA's ImageReward by 3.70 [3.42, 3.96] and full-frame MUSIQ by 20.05 [13.86, 24.68], worsening all ten metrics. `st@ds@d@def@ir` `st@ds@ci@def@ir` `st@ds@d@def@musiq` `st@ds@ci@def@musiq` `st@ds@sig@def`

## Not run

21. Qwen-Image transfer (E3) did not run: all 60 generation tasks hit CUDA out-of-memory beside a co-resident ~45 GiB process; no claim about other model families.

## Abstract-ready sentence (one option)

"A registered factorial on 187 held-out prompts shows that SEGA's global attention temperature carries its whole ImageReward gain over an NTK base (+0.594 [+0.506, +0.686]), while its RoPE base adds +0.027 [-0.003, +0.056]; on 337 pooled prompts the base-plus-temperature control is equivalent to released SEGA within preregistered margins (ImageReward +0.009 [-0.020, +0.041], PickScore -0.017 [-0.059, +0.026]), and on 150 fresh prompts the per-band allocation adds +1.56 [+0.43, +2.72] full-frame MUSIQ."
