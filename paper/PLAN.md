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
   but real full-resolution sharpness gain (MUSIQ about +2) and nothing on reward metrics.
2. Principled per-band laws do not help: four vision-science-derived controllers (Constancy RoPE from efficient coding and
   contrast constancy, its time-matched version, alias-count per-band temperature, SEGA plus time-matched correction)
   tie or lose to released SEGA under a pre-registered dev to held-out gate. Report as a negative result with mechanism
   analysis (why the gate stays shut, time-matched target, attention-gain transfer probe).
3. Checkpoint dependence: FLUX.1-dev breaks into a dot/halftone lattice at 4096^2 even under the OFFICIAL DyPE code;
   FLUX.1-Krea-dev (used by DyPE and SEGA) is clean. Results on one checkpoint do not transfer.
4. Paper-versus-code gap in SEGA: the published equations differ from the released code; as written they perform much
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
Tables: main held-out table (means with CIs), ablation table, compute/timing table.

## Quality bar
Professional academic look, attractive, clearly better than the figures in the reference papers the user named
(NVIDIA Nemotron and SOL technical reports, "Diagonal Distillation", "LoT"); look these up to calibrate, do not copy.
Consistent palette, one font family, consistent method colors and names across all figures and tables.
Prose: no em dashes, no section symbol, avoid "license", "warrant", "plainly". Captions lead with the finding.
Build PDF with ~/.local/bin/tectonic; inspect every page as rendered images before declaring done.
Deliver: paper PDF + standalone source zip that compiles (Overleaf-ready).
