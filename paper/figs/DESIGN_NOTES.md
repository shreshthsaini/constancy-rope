# Figure design brief (figv3, 2026-10-19)

Source of truth for every figure in `paper/figs/`. The shared module `crope_style.py` implements
this file (names, palette, markers, helpers, data access, bootstrap, image helpers); every script
imports it. The binding bar is the `gold-standard-paper` skill (LILIE, arXiv 2610.23653, and
MFlow, arXiv 2610.23854), sections 3, 4 and 6. `CRITIQUE_v2.md` holds the per-figure analysis that
led to this version; where it disagrees with this file, this file wins.

## The user's rule

Minimal text inside figures and diagrams: labels of 1 to 3 words, names, symbols and values,
never a sentence. Explanation goes in the caption. Counts of the text inside each figure are in the
table at the end and are measured with `count_words.py`.

## Palette (see PALETTE.md for tints and LaTeX names)

| Role | Arms | Color | Marker, line |
| :-- | :-- | :-- | :-- |
| Method hue | SEGA (released) | `#0068A8` | circle, solid |
| Component: base and temperature | SEGA base + temp. | `#009870` | triangle, solid |
| Component: per-band laws | Constancy-TM, SEGA + TM, Constancy, Alias-T | `#D06008` | diamond solid, down-triangle dash-dot, circle dotted, square dotted |
| Neutral gray | DyPE | `#8C8C8C` | square, dashed |
| Black | NTK baseline, all text | `#1A1A1A` | circle, dashed |
| Appendix only | SEGA (paper eqs.) | `#C878A0` | square, dashed |

Color logic: black is the baseline the gain is measured from, the method hue marks the released
method, green marks the component that carries most of its gain (base and global temperature),
orange marks the per-band laws we tested. In the method diagram the same hues name the knobs:
RoPE base neutral, temperature green, per-band gain and SEGA's sensor blue, the Constancy-TM
controller orange. At most four hues in any one figure. Method names are set in black; only marks
carry the hue. Diagram fills are 20% tints and table row shading 10% tints of the same hex values.
Heatmaps use white to the method hue; signed maps use method blue, white, law orange.

## Typography

- One sans face: Nimbus Sans (Helvetica metrics) in matplotlib, TeX Gyre Heros in TikZ. Math in
  math font.
- Sizes at final print size: tick labels 7 pt, axis titles and row labels 8 pt, panel tags 9 pt
  as "(a) Name" at the upper left, printed values 7 pt. Nothing below 6.6 pt anywhere.
- Panel tags only on multi-panel figures; plots carry no titles beyond a one-line metric name
  with its better-direction arrow.

## Plots

No top or right spines; light gray gridlines (`#E6E6E6`, 0.5 pt) behind the data; one shared
legend above the panel or row, no frame; gray band (`#F2F2F2`) for regions of interest; dashed
verticals for reference values (zero line of a paired difference against SEGA (released) is a
dashed line in the method hue); thin lines (about 1 pt) and small markers; paired 95% bootstrap
intervals as whiskers or bands.

## Photo panels

- Methods as rows with labels at the left (8 pt, SEGA (released) bold); same prompt and seed in
  every column.
- Zoom: 1 pt red (`#FF0000`) square box on the full frame and a 1 pt red frame on the enlarged crop;
  a second crop on the same frame uses cyan (`#00B0F0`). No halos, no letters.
- Photos embedded as JPEG (quality 94, 4:4:4) inside the vector PDF by `crope_style.jpegify`.
  Full frames reduced with Lanczos only; crops are native pixels, never resampled or sharpened.

## Size and layout

Every figure is built at its final print width (text width 5.50 in) and included so LaTeX scales
it by 1.0; the include width of each figure is listed below.

## Metric names and directions

ImageReward, PickScore, CLIP score (reward and alignment); MUSIQ and CLIP-IQA on the full frame and
on 1024-pixel crops (image quality); spectral log-RMSE, spectral slope error, tiling index at
1024 px (spectrum and structure; lower is better). Paired deltas are method minus released SEGA
unless a figure states otherwise.

## Statistics shown

Held-out rows 8 to 194 (n = 187), seed 0, paired with released SEGA on the same prompt and seed.
95% CIs are percentile intervals of 2,000 paired bootstrap resamples over prompts (seed 0), identical
to `scripts/heldout_analysis.py`. The seed-1 replication uses rows 8 to 62 (n = 55) with the same
bootstrap. Constancy (NTK base) has n = 41; Alias-T exists only on the eight dev prompts. Main-text
result figures show held-out results only.

## Text inside each figure (figv3)

Counted with `count_words.py` (PyMuPDF words). English words: two or more letters; math labels such
as ln P are counted apart; numerals are tick labels and printed values. Before: CRITIQUE_v2 counts.

| Figure | File | Print size (in) | Include | English words before | English words now | Numerals now | Main change |
| :-- | :-- | :-- | :-- | --: | --: | --: | :-- |
| F1 hero | f1_hero.pdf | 5.50 x 3.12 | \linewidth | 48 | 17 | 9 | LILIE-style hero: (a) paired-gain bars with CIs and printed values, (b) three-arm mosaic with red and cyan zoom boxes and native crops below |
| F2 method | f2_method.pdf | 5.50 x 2.17 | \linewidth | 77 | 17 | 0 | gold recipe: tinted rounded blocks, 0.6 pt orthogonal arrows, held-out photo with the key-query offset, measured step-14 gains; knobs colored as the arms that add them |
| F3 forest | f3_forest.pdf | 5.50 x 2.90 | \linewidth | 76 | 29 | 24 | legend, shading, group headers and printed values removed; neutral row labels; dashed blue SEGA (released) line; arrows in metric names |
| F4 spectrum | f8_spectrum_main.pdf | 2.75 x 2.00 | wrapfigure 0.5\linewidth | 17 | 11 | 12 | wrapfigure size; frameless two-row legend; gray 8 to 32 px band; dashed Nyquist line; dashed blue reference |
| F5 qualitative | f4_qualitative_main.pdf | 5.50 x 4.30 | \linewidth | 32 | 16 | 0 | methods as rows with labels at left; 2-word prompt tags in gray bands; red zoom box with its native crop beside each frame |
| A.1 qualitative | f4_qualitative.pdf | 5.50 x 3.10 | \linewidth | 39 | 16 | 0 | same design as F5 with red and cyan crops per frame |
| A.2 checkpoint | f5_checkpoint.pdf | 5.50 x 3.26 | \linewidth | 54 | 16 | 17 | panel tags; rotated row labels; column labels under images; red and cyan windows; peak ratios moved to the caption |
| A.3 spectrum | f8_spectrum.pdf | 5.50 x 2.45 | \linewidth | 45 | 20 | 25 | side-by-side panels; math-only y labels; one legend row; direction sentences removed |
| A.4 paper vs release | f6_sega_code.pdf | 5.50 x 4.73 | \linewidth | 64 | 33 | 34 | score table moved to f6_sega_code_table.tex; range labels removed; direct labels; red crop boxes |
| A zoom (controllers) | fA_method_zoom.pdf | 5.50 x 3.53 | \linewidth | new | 30 + equations | eq. | new LILIE A.1-style zoom: each controller as a box chain from x_t to g with its equations inside the panel |
| F.1 mechanism | f7_mechanism.pdf | 5.50 x 4.02 | \linewidth | 40 | 22 | 42 | in-figure sentence and title removed; slim vertical colorbar; panel tags |

Every figure went through at least three render-view-revise rounds at 160 dpi against the gold crops;
renders are in /scratch/09032/saini_2/tmp/figv3/. Caption text and the LaTeX handoff are in CAPTIONS.md.
