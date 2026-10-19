# Paper plan (source of truth for the paper agents), 2026-10-18

## Framing (non-negotiable)
Honest findings paper. NO state-of-the-art claim, NO claim that any proposed method beats released SEGA.
Every number must come from a file listed below and be reproducible by a script in `paper/figs/` or `paper/tables/`.
Working title (agents may improve): "Where Does the Gain Come From? A Controlled Study of Training-Free RoPE
Extrapolation in Diffusion Transformers". Venue style: NeurIPS-like single column, 9 pages main + appendix.

## Contributions (in order of strength)
1. Controlled decomposition of a strong training-free 4K method (released SEGA, arXiv 2605.22668) into RoPE base,
   global attention temperature, and per-band allocation. Held-out, paired, bootstrap CIs (187 prompts, seed 0; 55-prompt
   seed-1 replication). Finding: base + global temperature carry most of the gain; SEGA's spectral allocation adds a small
   full-frame MUSIQ gain (about +2) and no detected reward gain. Reduced excess fine-scale power accompanies that
   score change; neither observation establishes perceived sharpness or a causal mechanism for MUSIQ.
2. Principled per-band laws do not help: four vision-science-derived controllers (Constancy RoPE from efficient coding and
   contrast constancy, its time-matched version, alias-count per-band temperature, SEGA plus time-matched correction)
   tie or lose to released SEGA. The written dev-to-held-out gate began in campaign 2; the decomposition hypothesis
   came from campaign-1 development observations. Diagnostics suggest explanations, not proven quality mediators.
3. Checkpoint/guidance dependence: the original FLUX.1-dev versus Krea diagnostic changes guidance (3.5 versus 4.5).
   Official-code evidence is one dev prompt per checkpoint. Do not claim an isolated checkpoint effect until the
   complete matched-guidance matrix is verified; retain the older comparison's scope in captions.
4. Paper-versus-code gap in SEGA: our reading differs at each of the six compared stages; as implemented it performs
   worse and their static tilt destroys images (CLIP 13).
5. Evaluation and engineering pitfalls at 4K: 224-pixel reward models are blind to 4K detail; the diffusers default flow
   shift extrapolates mu to about 11.5 at 4096^2 and collapses the schedule (official repos use mu = 1.15); forced
   FlashAttention-2 takes 1.54x the time of cuDNN SDPA on GH200; one GB200 node is 2.01x faster than GH200.
6. Appendix: mechanism diagnostics CR14 to CR20 (see .agent/notes/*cr1* and STATE.md), only what is clearly established.

## Data sources (scratch = /scratch/09032/saini_2/constancy_rope)
- Held-out tables: scratch/eval_krea/heldout/{heldout_means.csv, heldout_paired_vs_sega_official.csv, heldout_per_prompt.csv}
  (includes main, mainsb (_sbase), main2 (_v2) roots). Seed 1: scratch/eval_krea/seed1/per_image.csv + eval_krea/ir/seed1.csv.
- Per-image metrics: scratch/eval_krea/{main,mainsb,main2,pilot,devsb,dev2_*}/per_image.csv; ImageReward scratch/eval_krea/ir/*.csv.
- Images (4096^2 PNG + JSON sidecars with per-step gains): scratch/out_krea/{main,mainsb,main2,pilot}/METHOD/4096x4096/.
  FLUX.1-dev (lattice regime): scratch/out/pilot/METHOD/4096x4096/ ; official DyPE diag: scratch/diag/official_dype_{dev,krea}_4096_*.png.
- Probe (attention-gain transfer): scratch/probe_krea/4096x4096*/mtf.csv.
- Calibration: scratch/calib/krea_profile.npz, krea_profile_tm.npz (time-resolved native spectra).
- Dev selection tables: .agent/notes/dev2/dev_table.csv; second-iteration report .agent/notes/main2/REPORT.md.
- CPU mechanism study of SEGA (forward-noised latents): ../research_reviews/sega_2605.22668_2026-10-14/{stats20.json,stats_4k.json,blur20.json}.
- Telemetry/timing: scratch/telemetry/*.csv, per-image `seconds` in JSON sidecars.
- Method definitions: crope/gain.py, crope/rope.py, crope/sega_official.py, .agent/notes/design.md, .agent/notes/alias_temp.md.
- Narrative history and verdicts: HANDOVER.md, .agent/notes/results.md.

## Figures (agents own `paper/figs/`; scripts beside outputs; vector PDF; orx_figstyle; audit clean)
F1 hero (page 1, full width): the decomposition claim at a glance, combining 4K crops with a compact quantitative panel.
F2 method diagram (TikZ or vector): attention logit as a Fourier series over offsets; where RoPE base, global temperature,
   and per-band gain act; where SEGA's sensor and the Constancy controller plug in.
F3 held-out forest plot: paired deltas vs released SEGA with 95% CIs for every method and metric; seed-1 replication marked.
F4 qualitative 4K comparison grid with zoom insets (same prompt/seed across methods; at least 3 prompts).
F5 checkpoint dependence: FLUX.1-dev lattice vs Krea under official DyPE, full frame + 1:1 crops.
F6 SEGA paper vs code: per-dimension gain over denoising time (heatmaps) + resulting images.
F7 mechanism: Constancy gate activity (late-profile vs time-matched) and the attention-gain transfer probe matrix.
F8 spectral error by octave per method (vs Aesthetic-4K references).
Promote F8's paired spectrum panel as `f8_spectrum_main.pdf` in the decomposition section; retain full spectrum appendix.
Tables: put the condensed paired table (`tab:main`, formerly Table 3) in the appendix alongside all-metric results.
Nominal intervals/colours remain visible; stars identify approximate Holm-adjusted centered-bootstrap tests within
each table's paired family. Main decomposition MUSIQ survives; weak CLIP/CLIP-IQA findings need the adjusted qualifier.

## Senior-review revision, 2026-10-19 UTC
- Finished paper prose must state measured evidence only. Commented STRENGTHEN insertion points mark the factorial,
  temperature sweep, fresh150, multiseed, matched-guidance, default-shift and Qwen analyses until each is complete.
- The dated preregistration amendment controls the new analysis. Do not infer equivalence from a zero-crossing CI.
  Keep original held-out diagnostics separate from fresh confirmation and report failed or unresolved hypotheses.
- MUSIQ audit: pyiqa 0.1.16 KonIQ10k, native RGB plus bicubic224/384 views,32x32 stride32 patches. Native4K has
  all16,384 patches and no cap;16,577 total before CLS. Four1024 crops are scored separately and averaged.
  Full-frame/crop differences also include context and scale weighting. This is quality scoring, not sharpness.
- Hero revision uses row32 (cabin), selected by caption subject before viewing that row; shared384 crops were chosen
  after viewing for common cabin/mountain content. Disclose this illustrative selection and crop centres in caption.
  Figure 1b uses two labelled dot-and-interval rows per metric: base plus temperature minus NTK,
  and released SEGA minus base plus temperature. Intervals are unshifted paired 95% CIs; ratios remain in the appendix.
- SEGA source pin is ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e; printed-law references are Sections4.1/4.2,
  equations3-7 and9. Record our latent/index-axis interpretation and that author confirmation was not obtained.
- One-prompt cross-checkpoint gain-probe sign change belongs only in the mechanism appendix, with guidance/code
  confounds explicit. Seed1 and partial Constancy rows are manifest prefixes, not random subsamples.
- Completed figure revisions: F1 absolute gains and shared-object crops, F2 separate illustrative traces,
  F3 direction shading and seed legend, F4 two-row main grid, A.2 gain scales and line profiles,
  F7 individual-prompt matrices, and the promoted spectrum panel with a Nyquist marker.
  Inspect the compiled pages at print size; update result-dependent figures only after verified new data.

## Camera-ready layout, 2026-10-19
- Sections 1 to 7 occupy nine pages; references start on page 10. Keep the forest and spectrum as Figures 3 and 4.
- The six-stage comparison and full controller equations are in Appendix E. Development diagnostic values,
  checkpoint figure, spectral-band details and pilot timings are in the corresponding appendices.
- Figure 2 is included at full text width so its smallest text remains above 6 pt.
- Table direction triangles and Holm stars use neutral ink; method colours retain their figure meanings.
- Camera-ready receipts and the independent raw-CSV number audit are in `paper/qa/`. Pending confirmation
  results remain the other process's responsibility; none were added during this pass.

## Quality bar
Professional academic look, attractive, clearly better than the figures in the reference papers the user named
(NVIDIA Nemotron and SOL technical reports, "Diagonal Distillation", "LoT"); look these up to calibrate, do not copy.
Consistent palette, one font family, consistent method colors and names across all figures and tables.
Prose: no em dashes, no section symbol, avoid "license", "warrant", "plainly". Captions lead with the finding.
Build PDF with ~/.local/bin/tectonic; inspect every page as rendered images before declaring done.
Deliver: paper PDF + standalone source zip that compiles (Overleaf-ready).
