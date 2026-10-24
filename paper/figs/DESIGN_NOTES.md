# Figure design brief (figv3 2026-10-19; v4 figure update 2026-10-24)

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
  as "(a) Name" at the upper left, printed values 7 pt, legends 7 pt. Nothing below 6.5 pt anywhere: TikZ
  math sub- and superscripts are declared at 6.5 pt (v4; they were 6.1 pt in F2 and F.1 before).
- Panel tags only on multi-panel figures; plots carry no titles beyond a one-line metric name
  with its better-direction arrow.

## Plots

No top or right spines; light gray gridlines (`#E6E6E6`, 0.5 pt) behind the data; one shared
legend above the panel or row, no frame; gray band (`#F2F2F2`) for regions of interest; dashed
verticals for reference values (zero line of a paired difference against SEGA (released) is a
dashed line in the method hue); thin lines (about 1 pt) and small markers; paired 95% bootstrap
intervals as whiskers or bands.

## Photo panels

- Appendix grids (A.2, A.4): methods as rows with labels at the left (8 pt, SEGA (released) bold); same
  prompt and seed in every column.
- Body grid (Figure 5, v4) and hero panel (b): arms as columns with names once at the bottom (8 pt, SEGA
  (released) bold, every line on a shared baseline), prompts as rows tagged at the left in a gray band (8 pt,
  rotated), and the native crop directly below its full frame at the same width (gold standard, Section 6).
  Columns come in palette pairs: NTK, DyPE | SEGA (released), SEGA base + temp. | Constancy-TM, SEGA + TM.
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

## Text inside each figure (v4, current paper numbering)

Counted with `count_words.py` (PyMuPDF words). English words: two or more letters; math labels such
as ln P are counted apart; numerals are tick labels and printed values. "figv3" is the count before v4 (for
F1 and F2 recounted with the same script; CRITIQUE_v2 counts are in the figv3 history of this file).

| Paper | File | Print size (in) | Include | English words figv3 | English words now | Numerals now | Main change |
| :-- | :-- | :-- | :-- | --: | --: | --: | :-- |
| Figure 1 hero | f1_hero.pdf | 5.50 x 3.12 | \linewidth | 17 | 18 | 13 | v4: solid black total line (SEGA (released) minus NTK) across each metric's bars, named "Total" once; a printed value the line would cut is lifted above it (MUSIQ) |
| Figure 2 method | f2_method.pdf | 5.50 x 2.17 | \linewidth | 17 (+ ln) | 19 (+ ln) | 0 | v4: ln g panel 72 to 98 bp wide (rotary panel 148 to 122 bp), outlined temperature bars, larger marks, three-glyph legend under the panel (circle SEGA, diamond Constancy-TM, outlined bar tau), subscripts 6.5 pt, gain arrow routed between label and legend |
| Figure 3 spectrum | f8_spectrum_main.pdf | 2.75 x 2.00 | wrapfigure 0.5, \linewidth inside | 11 | 11 | 12 | unchanged in v4 |
| Figure 4 confirmation | f4_spectrum.pdf | 2.75 x 1.83 | wrapfigure 0.5, \linewidth inside | 10 (fs_confirm.pdf, full width) | 13 | 7 | v4: the two reward panels at half width, about 1.5:1, one signed header with a rule over both panels; MUSIQ panel moved to the appendix version; ticks 7 pt at scale 1.0 |
| Figure 5 qualitative | f5_qualitative_body.pdf | 5.50 x 3.40 | \linewidth | new | 14 | 0 | v4: held-out rows 8 and 9 (first two held-out rows) by six arms; full frame above its native 384 px crop; A.2's crop boxes asserted; crops Flate-lossless and byte-checked |
| Figure A.1 forest | f3_forest.pdf | 5.50 x 2.90 | \linewidth | 29 | 29 | 24 | unchanged in v4 |
| Figure A.2 qualitative | f4_qualitative_main.pdf | 5.50 x 4.30 | \linewidth | 16 | 16 | 0 | unchanged in v4 |
| Figure A.3 spectrum | f8_spectrum.pdf | 5.50 x 2.45 | \linewidth | 20 | 20 | 25 | unchanged in v4 |
| Figure A.4 qualitative | f4_qualitative.pdf | 5.50 x 3.10 | \linewidth | 16 | 16 | 0 | unchanged in v4 |
| Figure A.5 paper vs release | f6_sega_code.pdf | 5.50 x 4.73 | \linewidth | 33 | 33 | 34 | unchanged in v4 |
| Figure A.6 checkpoint | f5_checkpoint.pdf | 5.50 x 3.26 | \linewidth | 16 | 16 | 17 | unchanged in v4 |
| Appendix C confirmation | fA_spectrum_full.pdf | 5.50 x 1.62 | \linewidth | new | 19 | 11 | v4: the former body Figure 4 with each sign in a header over its panels (reward: SEGA base + temp. minus SEGA (released); MUSIQ: the reverse), two-line row labels, wider MUSIQ panel |
| Figure F.1 controllers | fA_method_zoom.pdf | 5.50 x 3.53 | \linewidth | 30 + equations | 30 + equations | eq. | unchanged in v4 |
| Figure G.1 mechanism | f7_mechanism.pdf | 5.50 x 4.02 | \linewidth | 22 | 22 | 42 | unchanged in v4 |

File names `f4_spectrum.pdf` and `fA_spectrum_full.pdf` were fixed by the integration plan; both hold the
registered-confirmation figure (`fig:fs-confirm`), written by `f4_spectrum.py`, not the luminance spectrum.

v4 rounds (160 dpi renders and a scratch test page with the paper geometry and caption style, in
/scratch/09032/saini_2/tmp/tmp/claude-883311/-work-09032-saini-2-vista-Projects-Exploration/64a04f3f-4a9f-4722-a656-1099a206e63c/scratchpad,
session scratch, purge-eligible): Figure 5 six rounds (layout; gray tag bands;
shared label baseline; per-line labels after matplotlib anchored a multi-line baseline at its last line;
two tile reductions to keep the float with its caption under 0.45 of the text block). Figure 2 three rounds
(widened panel and legend; arrow rerouted off the legend; zero line moved off the arrow tip). Figure 4 two
rounds (audit: y title, header overflow). Figure 1 one round.

Every figure went through at least three render-view-revise rounds at 160 dpi against the gold crops;
renders are in /scratch/09032/saini_2/tmp/figv3/. Caption text and the LaTeX handoff are in CAPTIONS.md.
