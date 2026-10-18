# Figure design brief

Source of truth for every figure in `paper/figs/`. The shared module `crope_style.py` implements
this file (names, colors, data access, bootstrap, image helpers); every script imports it.

## Calibration against reference papers

Surveyed (figures viewed): Nemotron 3 Ultra (arXiv 2606.15007), Sol Video Inference Engine (2606.23743)
and Sol-Attn (2607.24027), Diagonal Distillation (2603.09488), LoT Level-of-Token Diffusion (2610.05816),
DyPE (2510.20766), UltraImage (2512.04504).

What they do well:
- Sol-Attn: disciplined palette (one accent for the method, one for the baseline, grey for the rest).
- LoT: the strongest hero, a mechanism map beside the output with the numbers placed under each panel.
- Nemotron: zoom insets linked to their source by thin boxes and leader lines.
- DyPE: color-coded dashed crop boxes tied to a crop grid; Diagonal Distillation draws its method
  with real intermediate data as nodes.

Where they fall short, and what we do instead:
- Text at 3.5 to 5.5 pt next to an 11 pt body, and mixed font families (Nemotron, LoT, Sol-Engine).
  Ours: one Helvetica-design face, 6.5 to 8 pt at print size, nothing below 5.5 pt.
- Numbers hidden in subcaptions or translucent overlays (Sol, Diagonal Distillation). Ours: every
  qualitative claim sits next to a measured number with its 95% CI, outside the image.
- Teasers with no crops (DyPE) or no baselines (UltraImage), heavy trapezoid connectors. Ours: native
  1:1 crops at identical locations for every method, hairline boxes, crop size stated.
- Default tab10 plots, no uncertainty, method not emphasized (DyPE). Ours: fixed semantic palette below,
  paired bootstrap CIs everywhere, filled marker when the CI excludes zero, direction of "better" drawn.
- Near-gutterless image grids without zooms (Diagonal Distillation). Ours: 2 pt gutters inside a method
  group, wider gaps between groups, method names once as column headers, no numbers painted on photos.
- Method diagrams with clip art. Ours: real data glyphs (RoPE wavelengths, band weights, the logit curve)
  in paper notation, 0.6 to 0.8 pt strokes.

## Typography

- One face everywhere: Nimbus Sans (Helvetica metrics), resolved by `orx_figstyle.use_style()`.
  The TikZ method diagram uses TeX Gyre Heros, the same Helvetica design, so diagram and plots match.
- Sizes at final print size: tick labels 7 pt, axis labels 7.5 to 8 pt, annotations and direct labels
  6.5 to 7 pt, panel letters 9 pt bold lower case (a, b, c). Nothing below 5.5 pt.
- Math in figures uses stixsans so symbols sit with the sans text. No titles on axes: captions carry the claim.
- Sentence case for labels; units in labels; arrows in labels state direction ("higher is better").

## Size and layout grid

- The paper is NeurIPS-like single column: every figure is built at `TEXT = 5.5 in` wide and included
  with `width=\linewidth` (scale 1.0 at 5.5 in; scales up cleanly if the class keeps 6.5 in).
- Outer margin 0; gutter between panels 0.10 to 0.14 in; panel letters sit above the top-left corner
  of their panel on a shared baseline.
- Photo panels: no frames except a 0.4 pt hairline in `RULE` grey around 1:1 crops; crop boxes on full
  frames drawn in white with a 1.2 pt dark halo so they read on any content; crop numbers in small caps
  style labels ("A", "B") instead of colored boxes, so box colors never collide with method colors.
- Photos are embedded as JPEG (quality 94, 4:4:4) inside the vector PDF by `crope_style.jpegify`.
  Full frames are reduced with Lanczos only; 1:1 crops are native pixels, never resampled or sharpened.

## Method names and colors (fixed across all figures and tables)

| key | display name | color | role |
|---|---|---|---|
| `ntk2` | NTK | stone `#9A9A9A` | RoPE base only (NTK rule SEGA builds on), no temperature |
| `dype` | DyPE | teal `#2E9C85` | external baseline, official recipe |
| `sega_official` | SEGA (released) | cobalt `#1F4E8C` | reference for every paired delta |
| `temp_only_sbase` | SEGA base + temp. | sky `#7EA9D9` | SEGA's RoPE base and global temperature, no per-band allocation |
| `constancy` | Constancy | saffron `#E1A21B` | efficient-coding gate, late-profile target, NTK base |
| `constancy_tm_sbase` | Constancy-TM | terracotta `#D35F3A` | time-matched target on SEGA base + temp. |
| `sega_ctm_v2` | SEGA + TM | plum `#A2467F` | released SEGA allocation times time-matched correction |
| `alias_temp` | Alias-T | sienna `#8A5A33` | alias-count per-band temperature (dev only, failed gate) |
| `sega` | SEGA (paper eqs.) | oxblood `#9E2A2B` | SEGA as written in the paper equations |

Color logic: blues are the SEGA lineage (darker = more of SEGA), warm hues are the four principled
per-band rules, grey and teal are baselines. Reference lines and zero lines are `RULE` grey.
Direct labels are preferred over legends; markers: circles for baselines and SEGA, diamonds for the
principled rules, so the grouping survives greyscale.

Neutral ink: text `#1F2329`, secondary text `#5B6270`, hairlines `#C9CDD3`, panel fill `#F3F4F6`.
Sequential maps: `cividis`-like light-to-cobalt for gains; diverging `RdBu_r` centered at zero for
signed responses (elasticities, deltas).

## Metric names and directions

ImageReward, PickScore, CLIP score (reward and alignment); MUSIQ and CLIP-IQA on the full frame and on
1024-pixel crops (image quality); spectral log-RMSE, spectral slope error, tiling index at 1024 px
(spectrum and structure; lower is better). Paired deltas are always method minus released SEGA, with
the better direction drawn as an arrow beside each metric.

## Statistics shown

Held-out rows 8 to 194 (n = 187), seed 0, paired with released SEGA on the same prompt and seed.
95% CIs are percentile intervals of 2,000 paired bootstrap resamples over prompts (seed 0), identical
to `scripts/heldout_analysis.py`. The seed-1 replication uses rows 8 to 62 (n = 55) with the same
bootstrap. Constancy (NTK base) has n = 41; Alias-T exists only on the eight dev prompts.
