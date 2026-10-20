# Figure critique v2

Date: 2026-10-19. Read with `STYLE_REFERENCE.md` (same folder): rules R1 to R10 in its consolidated guide are cited here by number. The reference review (same date) added the two user-named papers LILIE (arXiv 2610.23653) and MFlow (arXiv 2610.23854) as primary references next to SEGA and AReA; where they or the `gold-standard-paper` skill set a higher bar, the specs below follow them.

**Standing instruction from the user.** Whenever we write a paper, its figures meet or exceed the bar these reference papers set, and the text inside figures stays minimal: names, symbols and axis numbers, never a sentence. Explanation goes in the caption.

Reviewed: `paper/dist/constancy_rope_paper.pdf` (25 pages, rendered at 160 dpi) and every standalone figure PDF that the manuscript includes.

Fixed constraints: the same arms, metrics, prompts, seeds, intervals and disclosures stay. The main text stays held-out only, and no change below adds a claim. Text deleted from a figure is either already in its caption or moves there with the wording given below. Sizes are print sizes; the text width is 5.50 in and the body text is 10 pt.

How words were counted: every text token extracted from the vector figure PDF with PyMuPDF (`get_text("words")`). Tokens fall into three groups: English words (two or more letters), numerals (tick labels and printed values), and symbols or single letters (math symbols, panel letters, crop letters). None of our photographs contain baked-in text. For targets, math labels such as "ln" do not count as English words. Caption caps (R9) count whitespace tokens of the compiled caption including the "Figure N:" label, the method the reference analysts used.

## Summary

| Fig. | File | Print size (in) | English words | Numerals | Symbols, letters | Caption words | Target English words inside | Caption cap (R9, tokens) |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |
| F1 hero | f1_hero.pdf | 5.50 x 3.57 | 48 | 24 (8 printed values) | 13 | 84 | 18 | 95 |
| F2 method | f2_method.pdf (TikZ) | 5.50 x 2.76 | 77 | 16 | 42 | 123 | 10 (plus about 17 math labels) | 110 |
| F3 forest | f3_forest.pdf | 5.50 x 3.34 | 76 | 31 (6 printed values) | 21 | 72 | 30 | 110 |
| F4 spectrum | f8_spectrum_main.pdf | 5.50 x 1.92 | 17 | 13 | 3 | 75 | 14 | 110 |
| F5 qualitative | f4_qualitative_main.pdf | 5.50 x 3.26 | 32 | 3 | 5 | 97 | 21 | 110 |
| A.1 qualitative | f4_qualitative.pdf | 5.23 x 7.14 | 39 | 4 | 7 | 87 | 25 | 140 |
| A.2 checkpoint | f5_checkpoint.pdf | 5.50 x 3.26 | 54 | 29 | 9 | 99 | 18 | 140 |
| A.3 spectrum | f8_spectrum.pdf | 4.84 x 2.98 | 45 | 21 | 14 | 88 | 19 | 140 |
| A.4 paper vs code | f6_sega_code.pdf | 5.50 x 5.60 | 64 | 62 (26 printed values) | 14 | 127 | 26 | 140 |
| F.1 mechanism | f7_mechanism.pdf | 5.50 x 4.52 | 40 | 43 | 7 | 128 | 18 | 140 |

Main-text figures currently carry 250 English words inside the artwork; the targets total 93. Before the reference review they totalled 74; the difference is the panel names and prompt excerpts that the gold-standard papers use (R5, R6). The cuts come from sentences, qualifiers, legends, repeated axis words and printed values, not from data.

## Cross-cutting problems (fix once, in `crope_style.py` where possible)

1. Sentences and qualifiers inside the artwork (R1). Examples: "native 384 px crops, identical coordinates" (F1), "spectrum of embedded x_t / standardized band energies" (F2), "seed 1: base + temp. and Constancy-TM only, n = 55" (F3), "A: highest minimum normalized detail across all arms; ..." (F5), "Late-profile correction is exactly 0 in all 8 prompts ..." (F.1). Every one of these belongs in the caption, and most are already there.
2. Printed values on plots (R4). F1 prints 8 values, F3 prints 6 off-axis values (one pair collides: "+0.216 ▶ −0.035" in the Log-RMSE and Slope error panels), A.4 prints a 26-value table under its photos. The intervals already show the numbers, and Tables B.1 and B.2 hold the exact values.
3. Too many simultaneous encodings in F3: hue (method), marker shape (rule family), fill (seed), opacity (CI excludes zero), background shading (better side), edge triangles (off-axis). The new F3 drops the better-side shading (the direction moves to an arrow in each metric title, R4) and the legend; hue, marker shape, hollow (seed 1), pale (interval includes 0) and the edge triangles stay, and the caption defines them once.
4. Legends and headers that restate the caption: F3 legend (2 lines, 21 words), F3 group headers (4 headers, 11 words), F1 "Paired changes in score (95% CI)", F.1 "Width-axis response: 3 prompts, single seed".
5. Decoration that adds color without data: six colored rule bars above the grid columns (F5, A.1), the grey bands behind thumbnails (F5, A.1), three pastel highlight boxes in F2. Diagrams carry at most 3 hue families and plots at most 4, plus greys (R3).
6. Two-line labels where one line suffices: F1 column headers ("SEGA base + temp." over "+ base, temperature"), A.2 row labels ("DyPE / official code, / seed 42").
7. Panel letters: replace the bare bold "a" with "(a) Name" (9 pt bold, name of 1 to 3 words) at the top-left outside the panel, in F1, F2, A.2, A.3, A.4 and F.1 only (R5); F3, F4, F5 and A.1 carry no letters. No plot titles beyond a one-line metric name.
8. Type below the bar: tick labels and annotations at 6.5 pt (F1, F3, F4). Every glyph is now at least 7 pt (R2), as in BudgetPix and the gold-standard papers.
9. Hero below the abstract, and crop boxes in white with dark halos plus letters. Every reference at our text width that has a hero puts it above the abstract (SEGA, AReA, BudgetPix, MFlow), and every reference crop box is one saturated colour with a same-colour frame on the crop (R6, R7).

Global edits to `crope_style.py`: add `MAX_TICKS = 3` for small multiples, tick labels at 7 pt, and a helper `edge_marker(ax, side)` that draws the off-axis triangle without a value. Change `panel_tag` to print "(a) Name" (9 pt bold) instead of a bare letter. Add `GRID_LABEL = dict(fontsize=8, color=INK)` for image-grid method labels placed under the bottom row, with SEGA (released) in bold, and drop the colored rule bars. Add `CROP_COLORS = ("#FF0000", "#00B0F0")` with helpers `crop_box(ax, xywh, color)` and `crop_frame(ax, color)` (1.0 pt, square corners, no halo, no letter), and `prompt_band(fig, rect, text)` (fill `PANEL` `#F3F4F6`, 0.14 in tall, italic 8 pt `INK2`). Add `METRIC_ARROW` (↑ for ImageReward, PickScore, CLIP, MUSIQ and CLIP-IQA; ↓ for log-RMSE, slope error and tiling). Keep `TEXT_TINT` for direct labels on plots only. Update the matching lines of `DESIGN_NOTES.md` (crop boxes, panel letters, type sizes) in the same commit.

## F1 hero (page 1, `f1_hero.py`)

Current: 5.50 x 3.57 in, below the abstract. Panel a has a 1.6 in full frame with a two-line header, a 2 x 3 grid of 384 px crops (about 0.88 in each) with two-line column headers, and a five-line text block under the full frame. Panel b has four mini forest plots, each with metric title, two labelled rows and printed values. Text: 48 English words, 24 numerals, 13 letters.

What reads as noisy:
- Five text blocks compete before the eye reaches the crops: the full-frame header, three two-line column headers, and the "A: cabin / B: mountain / native 384 px crops, identical coordinates" block.
- Panel b repeats "base + temperature" and "allocation" four times and prints eight values; two values sit on the zero line ("−0.026", "+0.11").
- The photos occupy 55 percent of the height, but the crops are smaller than the full frame, so the detail the paper is about looks small.

Redesign spec:
- Size 5.50 x 3.20 in, full text width. Placement (R7): in `main.tex`, move `\input{sec/hero}` from after `\input{sec/abstract}` to directly after `\maketitle`, so the figure sits between the author block and the abstract, as in SEGA, AReA, BudgetPix and MFlow. Keep the minipage form of `sec/hero.tex`.
- Vertical layout from the top: a 0.14 in header row, panel a (2.14 in), a 0.12 in gap, panel b (0.80 in).
- Panel a, "(a) Native crops":
  - Full frame (SEGA (released), row 32) at 2.14 x 2.14 in, left-aligned, its top level with the top of the crop grid.
  - Crop boxes on the full frame (R6): cabin in red `#FF0000`, mountain in cyan `#00B0F0`, 1.0 pt square outlines at the existing coordinates; no halo and no letters.
  - Crops: a 2 x 3 grid of native 384² crops, each 1.055 x 1.055 in, 0.03 in gutters, starting 0.135 in right of the full frame (2.14 + 0.135 + 3 x 1.055 + 2 x 0.03 = 5.50 in). Row 1 is the cabin, row 2 the mountain. Each crop carries a 1.0 pt frame inside its edge in its row's colour.
  - Column headers in the header row, one line each, 8 pt, neutral ink `INK`, centred over the crop columns: "NTK", "SEGA base + temp.", "SEGA (released)", the last in bold. F1 is the one grid whose labels sit above the images, because panel b fills the space below.
  - Delete the full-frame header ("Full frame / SEGA (released), 4096 px"), the second header lines ("base only", "+ base, temperature", "+ per-band allocation"), the "A" and "B" letters and the five-line block under the full frame.
- Panel b, "(b) Paired gains", 0.80 in tall:
  - One row of four mini panels, 1.10 in wide each with 0.12 in gaps, after a 0.74 in label column (0.74 + 4 x 1.10 + 3 x 0.12 = 5.50 in).
  - Row labels in the label column, 8 pt: "base + temp." in sky text tint `#3F73B0` and "allocation" in cobalt `#1F4E8C`. Do not repeat them in the other panels.
  - Metric names above each mini panel, 8 pt, one line: "ImageReward ↑", "PickScore ↑", "MUSIQ ↑", "MUSIQ, crops ↑".
  - Marks: dot 3.5 pt in diameter, whisker 1.0 pt without caps, zero line 0.5 pt `RULE`. Three x ticks per panel (0 and two round values) at 7 pt; no y axis.
  - Delete all eight printed values and the "Paired changes in score (95% CI)" header.
- Panel letters with names (R5): "(a) Native crops" and "(b) Paired gains", 9 pt bold, top-left outside each panel.
- English words inside after the change: 18 (4 in panel names, 6 in headers, 3 in row labels, 5 in metric names).
- Replace the caption in `sec/hero.tex` with this text (91 tokens with the label; keep the existing macros for arm names and `\N{ho@nprompts}`):
  "**Where SEGA's gain comes from.** SEGA's base and temperature carry most of its gain; allocation adds full-frame MUSIQ. (a) Held-out row 32, seed 0, chosen by its cabin caption before viewing. *Left:* SEGA (released) at 4096²; *right:* native 384² cabin (red) and mountain (cyan) crops, fixed after viewing at (2048, 3200) and (2048, 1536), identical for every arm. (b) Gain of SEGA base + temp. over NTK and of SEGA (released) over SEGA base + temp.: means and nominal 95% paired bootstrap intervals over 187 held-out prompts, seed 0."
  The resample count (2,000) already appears in `sec/protocol.tex`, line 13. The draft version of this caption dropped "seed 0" from (b); it stays.
- Acceptance: no text inside the image block except the three headers; no number in panel b except tick labels; no glyph below 7 pt; caption at most 95 tokens.

## F2 method diagram (page 3, `f2_method.tex`, data from `f2_method_data.py`)

Current: 5.50 x 2.76 in, three panels. Panel a shows the token grid, query, key and offset, a "RoPE base b → b′" tag and a band-8 inset. Panel b shows the equation header, four band waves, the gain lollipops, the temperature tag, Σ and three logit traces. Panel c shows two controller boxes, each with a two-line description. Text: 77 English words, 16 numerals, 42 symbols. There are three pastel highlight fills (grey, blue, pink), and tick labels sit on three schematic plots.

What reads as noisy:
- Every element carries a descriptive phrase, so the diagram reads like a paragraph: "sets rotary frequencies", "equal gain on every band", "weights each band", "two alternative controllers", "native target, matched noise", "gated spectral error".
- Tick labels (−255, 0, 255; 0, 64, 256) on schematics imply measured axes that the caption then disclaims.
- The 25-token equation header repeats Eq. (1), which sits just below on the same page.

Redesign spec (two panels read left to right, R8):
- Size 5.50 x 2.00 in. Columns of 1.15, 2.10 and 1.75 in with 0.20 in gutters; a 0.45 in controller strip under columns 2 and 3. Lay everything out on an 8 pt grid. Panel (a) is column 1; panel (b) holds columns 2 and 3 and the controller strip (a method figure has at most two panels, guide rule 14).
- Panel letters with names (R5), 9 pt bold, top-left: "(a) Query-key offset" over column 1 and "(b) Per-band gains" over column 2. The working names "where the knob acts", "bands and gains" and "logit" below are not printed.
- Column 1 ("where the knob acts"):
  - Top: the token grid as now, the key a hollow circle, the query a filled circle and the arrow between them; its only label is "Δ" (math). Under the grid, place the final image of the held-out sample that `f2_method_data.py` uses for the gains, Lanczos-reduced to the grid's size and drawn at 35% opacity, so a real image enters the diagram (R8; LILIE Figs 3 and 4).
  - Delete "4096² image, 256² tokens", "native 64²" and its dashed box, "key p′", "query p", "Δ_w", "Δ_h", the right-angle guide and "Δ = p − p′".
  - Bottom: the band-8 inset with only the two curves (dashed b, solid b′) and their end labels "b" and "b′". Delete its ticks, the "offset (tokens)" label, the grey 0 to 64 shading and "band j = 8, one axis".
  - Above the inset, the tag "RoPE base" at 8 pt in `INK2`, with no fill.
- Column 2 ("bands and gains"):
  - The four waves stay, labelled θ6, θ8, θ10, θ12 in math at 8 pt. Delete their tick labels and the "offset Δ (tokens)" label; keep a single "Δ" under the axis.
  - The gain column sits right of the waves, headed "ln g_j" (math). Keep the sky band for ln τ, cobalt dots for SEGA and terracotta diamonds for Constancy-TM.
  - The temperature tag reads "temperature τ" at 8 pt in sky tint `#3F73B0`, with no fill box. Delete "global" and "equal gain on every band".
  - Delete "4 of 28 bands, width axis" and "log gain".
- Column 3 ("logit"):
  - The Σ node, then the three traces stacked as now, labelled at their left ends in math only: "b", "b′", "b′, τ". A single "ℓ(Δ)" label sits above the stack.
  - Delete "four-band logit slice", "native base", "released base", "base ... + temperature", the tick labels and the "offset Δ (tokens)" label.
- Controller strip (part of panel b):
  - Two rounded rectangles, corner radius 2 pt, 18 pt tall, 0.8 pt borders in cobalt (left) and terracotta (right), each filled with a 20% tint of its border colour (R3, R8). Titles at 8 pt in the border colour, one line each: "SEGA sensor" and "Constancy-TM".
  - One input arrow labelled "x_t" enters each box; one arrow per box rises into the gain column, labelled "g_j". All arrows 0.6 pt, straight or with one elbow, small filled heads.
  - Delete both description lines in each box, "two alternative controllers", "per-band gain g_j" and its pink box, and "weights each band".
- Delete the equation header; the caption points to Eq. (1).
- Colours: grey (base), sky and cobalt (one blue family), terracotta: two hue families plus grey (R3). No fills except the sky band and the two 20% tints.
- Fonts: TeX Gyre Heros 8 pt for tags and panel names, math labels 7 to 8 pt; nothing below 7 pt.
- Label tokens inside after the change: about 27 (10 English words: 4 in panel names and 6 in tags; about 17 math labels), inside R1's 15 to 30.
- Replace the caption in `sec/methods.tex` with this text (110 tokens with the label; use `\cref{app:laws}` and `\cref{fig:papercode,fig:mechanism}` for the pointers):
  "**The three knobs.** The RoPE base sets frequencies, temperature scales all terms, and per-band gains reweight each term. (a) Query, key and offset Δ on the 256² token grid over a held-out 4096² sample; inset: band 8 under the native base b (dashed) and SEGA's released base b′. (b) Four of the 28 width-axis bands of Eq. (1) and their total log gains (sky: ln τ; circles: SEGA (released); diamonds: Constancy-TM) from that sample at step 14; logit traces (unit amplitudes, zero phases) are illustrative, not measured. Below, two controllers set g_j from x_t; equations in Appendix C, gains per pair and step in Figures A.4 and F.1."
- Acceptance: no tick labels anywhere; no label longer than three words; no box with more than two words; caption at most 110 tokens.

## F3 forest plot (page 6, `f3_forest.py`)

Current: 5.50 x 3.34 in, 10 panels in two rows of five. Text: 76 English words, 31 numerals, 21 symbols. There are 4 group headers, 10 metric titles, 10 "better →" labels, "Method" and "Method − SEGA" twice each, a 5-entry two-line legend and 6 printed off-axis values.

What reads as noisy:
- The 10 "better" labels and the grey better-side shading say the same thing twice.
- The legend repeats caption content and adds 21 words under the plot.
- The off-axis value labels collide ("+0.216 ▶ −0.035").
- The group headers ("Reward and alignment", "Image quality: MUSIQ", "Image quality: CLIP-IQA", "Spectrum and structure") add a third text level above the metric titles.

Redesign spec:
- Size 5.50 x 2.95 in. Same 10 panels, same order, same arms, same seed-1 hollow markers, same filled versus pale significance coding. No panel letters (R5).
- Delete:
  - all four group headers;
  - both "Method" headers and both "Method − SEGA" footers;
  - all ten "better →" labels and the grey better-side shading; the direction moves into the metric titles (R4: one direction cue, never two);
  - the legend (both lines);
  - all six printed off-axis values. The edge triangles stay at the axis limit, at 4 pt.
- Metric titles, one line each at 7 pt (R2 allows 7 pt titles with five or more panels per row), with the better direction: "ImageReward ↑", "PickScore ↑", "CLIP ↑", "MUSIQ ↑", "MUSIQ, crops ↑", "CLIP-IQA ↑", "CLIP-IQA, crops ↑", "Log-RMSE ↓", "Slope error ↓", "Tiling ↓".
- Method labels stay at the left of each panel row, 7 pt, in the method text colours; they are the figure's only legend (R4).
- x ticks: three per panel (0 and two round values) at 7 pt; the zero line is 0.6 pt `INK2`.
- English words inside after the change: about 30 (16 method-label words, 14 metric-title words).
- Replace the caption in `sec/results_decomp.tex` (`fig:forest`) with this text (109 tokens with the label):
  "**Per-band laws against released SEGA.** No per-band law improves on SEGA (released) in any reward or image-quality metric. Paired held-out differences from SEGA (released), rows 8–194, seed 0 (n = 187), with 95% bootstrap intervals; pale markers: the interval includes 0. Hollow markers: seed-1 replication on rows 8–62 (n = 55) for SEGA base + temp. and Constancy-TM; SEGA + TM has none. Panels group reward and alignment, image quality (full frames and 1024-pixel crops), and spectrum and structure; arrows mark the better direction; triangles mark NTK beyond the axis limit (Table B.1). Isolated gains, such as crop MUSIQ and CLIP-IQA for DyPE, come with losses elsewhere."
- Acceptance: no text overlaps another element; no number on the plot except tick labels; caption at most 110 tokens.

## F4 paired spectrum (page 6, main panel of `f8_spectrum.py`, exported as `f8_spectrum_main.pdf`)

Current: 5.50 x 1.92 in, one line panel with 5 series, a two-row legend on top, a shaded 8 to 32 px band with its label, a dotted Nyquist line labelled "224-view Nyquist", and light gridlines. Text: 17 English words, 13 numerals. This is the cleanest figure in the paper.

What reads as noisy: the two-row legend takes 0.35 in above the plot; "224-view Nyquist" is jargon inside the plot area; the zero line is the reference arm but is not marked as such.

Redesign spec:
- Size 2.75 x 2.00 in, set as a wrapfigure at `0.5\linewidth` on the right of the paragraph that first cites it, with at least 8 text lines beside it (R10: the gold-standard papers wrap single plots; LILIE wraps five figures). No panel letter (R5).
- Legend: two unframed rows above the plot, 8 pt, 0.25 in line handles with markers: "NTK", "DyPE", "SEGA base + temp." in row 1 and "Constancy-TM", "SEGA + TM" in row 2. No direct labels.
- Axis labels at 8 pt: x "Wavelength (px)"; y "Δ ln P" in math. The zero line names the reference instead: write "SEGA (released)" at 7 pt in `INK2` directly above it at the left edge.
- Delete the "224-view Nyquist" text; keep the dotted line. Keep "8 to 32 px" above the shaded band, at 7 pt.
- Ticks: keep the six x ticks (4 to 4096 px) and the five y ticks, at 7 pt. Remove the vertical gridlines; keep the horizontal ones at 0.4 pt `GRIDC`.
- Raise the CI band opacity from about 0.10 to 0.18 so it is visible in print.
- English words inside after the change: 14 (8 in the legend, 2 in the x label, 2 in "8 to 32 px", 2 in "SEGA (released)").
- Caption (`fig:spectrum_main`): put the bold lead "**Paired luminance spectra.**" before the current first sentence and set that sentence in regular weight. Replace "lies above the marked Nyquist frequency of a 224-pixel view." with "lies above the Nyquist frequency of a 224-pixel view (dotted line)." The caption then has about 80 tokens with the label.
- Acceptance: legend in two rows above the plot; no text inside the plot area except "8 to 32 px" and "SEGA (released)"; caption at most 110 tokens.

## F5 qualitative grid (page 7, `f4_qualitative.py`, main variant)

Current: 5.50 x 3.26 in. Six arm columns are grouped in pairs, with colored headers and colored rule bars. Each prompt gets a row of small full-frame thumbnails (about 0.62 in) on a grey band, then a row of 384 px crops (about 0.86 in). The rotated row labels hold the row ID and a truncated prompt, and a footnote line sits under the grid. Text: 32 English words.

What reads as noisy:
- The thumbnails float on grey bands at two-thirds of the crop size, so the eye reads two different grids.
- The rotated, truncated prompts ("A stone bridge with arched spans…") and the dataset row IDs ("Row 8") are internal bookkeeping.
- The footnote ("A: highest minimum normalized detail across all arms; ...") repeats the caption.
- The six colored rule bars add six hues to a photo figure.

Redesign spec:
- Size 5.50 x 4.05 in. Six columns of 0.868 in square tiles, 0.03 in gutters inside a pair and 0.10 in between pairs (6 x 0.868 + 3 x 0.03 + 2 x 0.10 = 5.50 in). No borders. No panel letters (R5).
- Per prompt, from the top: a full-width prompt band 0.14 in tall (fill `PANEL` `#F3F4F6`) holding the excerpt in italic 8 pt `INK2`, left-aligned 0.04 in from its edge; the full-frame row (Lanczos-reduced); 0.03 in; the native 384² crop row. 0.10 in between the two prompts. Excerpts: "A stone bridge with arched spans…" (row 8) and "A panoramic view of Paris…" (row 9).
- Crop: a red `#FF0000` 1.0 pt square box on each full frame at the crop location; the crop tile directly below carries a 1.0 pt red frame inside its edge (R6, as MFlow Figs 22 and 23). No halo, no letter.
- Method labels under the bottom row (R6): one line each, 8 pt, neutral ink `INK`, centred under each column with the baseline 0.12 in below the last tile, "SEGA (released)" in bold. Order unchanged: NTK, DyPE | SEGA (released), SEGA base + temp. | Constancy-TM, SEGA + TM.
- Delete the coloured rule bars, the grey bands behind the old thumbnails, the crop letter "A", the rotated row labels with row IDs and prompts, and the footnote line.
- English words inside after the change: 21 (10 in method labels, 11 in the two prompt excerpts).
- Caption (`fig:qual_main`): put the bold lead "**Held-out samples at 4096².**" before the current first sentence and set that sentence in regular weight. Replace "The first two held-out prompts (rows 8 and 9), seed 0, with a full-frame thumbnail and an unaltered 384-pixel crop for each arm." with "Held-out rows 8 (top) and 9 (bottom), seed 0; each full frame sits above its unaltered 384-pixel crop (red box)." Keep the crop-selection disclosure and the remaining sentences verbatim. The caption then has about 98 tokens with the label.
- Acceptance: a 4 x 6 block of equal squares; no text on or between images except the two prompt bands; labels only under the bottom row; caption at most 110 tokens.

## A.1 qualitative grid, appendix (page 13, `f4_qualitative.py`, appendix variant)

Current: 5.23 x 7.14 in (0.95 linewidth), three prompts, each with a thumbnail row and two crop rows (A, B). It has the same headers, bars, grey bands, rotated labels and footnote as F5. Text: 39 English words.

Redesign spec: apply every F5 change, with these differences.
- Build at 4.85 x 7.78 in and include with `width=0.88\linewidth` (R10: an appendix grid that must fit one page may be narrower). Tiles 0.76 in; gutters 0.03 in inside a pair and 0.10 in between pairs (6 x 0.76 + 3 x 0.03 + 2 x 0.10 = 4.85 in). The figure keeps its page to itself, as now.
- Per prompt: the 0.14 in prompt band, the full frame with two 1.0 pt boxes (crop A red `#FF0000`, crop B cyan `#00B0F0`), crop A framed in red, crop B framed in cyan; 0.03 in between rows and 0.10 in between prompts. Height: 3 x (0.14 + 3 x 0.76 + 2 x 0.03) + 2 x 0.10 + 0.14 label row = 7.78 in. Excerpts: "A stone bridge with arched spans…", "A panoramic view of Paris…", "A cluttered artist's corner…".
- Delete the "A" and "B" letter column and the letters on the boxes; colour pairs each box with its crop.
- English words inside after the change: 25 (10 in method labels, 15 in prompt excerpts).
- Caption (`fig:qual`): put the bold lead "**Three held-out rows at 4096².**" before the current first sentence and set that sentence in regular weight. Replace "Held-out rows 8 to 10, the first three held-out rows," with "Held-out rows 8, 9 and 10 (top to bottom), the first three held-out rows," and replace "(A: highest-detail tile of the across-arm mean luminance; B: image centre)" with "(red: highest-detail tile of the across-arm mean luminance; cyan: image centre)". The caption then has about 97 tokens with the label.

## A.2 checkpoint diagnostic (page 14, `f5_checkpoint.py`)

Current: 5.50 x 3.26 in. Panel a has three rows (DyPE official code seed 42; DyPE our pipeline seed 0; SEGA released our pipeline seed 0) by two checkpoints, each a full frame plus two stacked 192 px crops; it has three-line row labels and a footnote. Panel b has three whitened spectra with in-plot annotations ("period ≈ 25 px", "peak ratio 36× (this pair only)", "8 prompts: peak ratio 12× [7, 20]", "8 prompts: peak ratio 5× [2, 11]"). Text: 54 English words, 29 numerals.

What reads as noisy: the three-line row labels, the footnote, and the peak-ratio sentences inside the spectra.

Redesign spec:
- Size unchanged (5.50 x 3.26 in).
- Panel letters with names (R5): "(a) Samples" over the image block and "(b) Whitened spectra" over the spectra.
- Row labels: rotated 90°, one line each, 8 pt, neutral ink `INK`, centred on their row in a 0.18 in strip at the left: "DyPE, official", "DyPE", "SEGA (released)". Delete the method-colour rules along the rows. Add the freed 0.44 in to the width of panel b.
- Column labels under the bottom image row (R6), one line each, 8 pt `INK`: "FLUX.1-dev" and "FLUX.1-Krea-dev". Delete "guidance 3.5" and "guidance 4.5" from the artwork; the caption already states them.
- Crop windows: A as a red `#FF0000` and B as a cyan `#00B0F0` 1.0 pt box on each full frame; the two stacked 192 px crops get 1.0 pt frames in the same colours. Delete the letters and the footnote line ("A, B: native 192 px windows ...").
- Panel b:
  - Each peak marker keeps only its period, "25 px" (rows 1 and 2) and "29 px" (row 3), at 7 pt, replacing "period ≈ 25 px".
  - Delete the three peak-ratio annotations; they move to the caption.
  - Direct labels on the top spectrum panel only, 8 pt: "FLUX.1-dev" on the black line and "Krea" on the grey line. Delete the legend box.
- English words inside after the change: about 18.
- Caption (`fig:checkpoint`): put the bold lead "**Lattice and checkpoint check.**" before the current first sentence and set that sentence in regular weight. Replace "A and B are native 192-pixel windows at the same place in every image." with "red and cyan boxes mark native 192-pixel windows at the same place in every image." After the sentence ending "does not in these samples." insert "Peak ratios: 36× for the official-code pair; over eight prompts in our pipeline, 12× [7, 20] for DyPE and 5× [2, 11] for SEGA (released) (geometric means, 95% bootstrap intervals)." Before committing, confirm which arm owns each ratio from the `print(col["key"], txt, ...)` line of `f5_checkpoint.py`. The caption then has about 134 tokens with the label.

## A.3 spectrum, appendix (page 14, `f8_spectrum.py`)

Current: 4.84 x 2.98 in, two stacked line panels with a 3 x 2 legend. The in-plot text reads "↑ more power than Aesthetic-4K (195 images)", "↓ less power", "↑ more power than SEGA (released)" and "↓ less power", plus a rotated "224-view Nyquist". Text: 45 English words.

Redesign spec:
- Size 5.50 x 2.98 in, built at full width and included at `\linewidth` (R10; it is now 4.84 in).
- Panel letters with names (R5): "(a) Error vs. reference" and "(b) Paired difference".
- Delete the four direction sentences. y labels in math only: (a) "ln P − ln P_ref", (b) "ln P − ln P_SEGA". The caption states the sign.
- Delete the rotated "224-view Nyquist" text; keep the dotted line.
- Legend: one unframed row of 6 entries above panel a, 8 pt, 0.20 in handles, 0.10 in between entries, in the order NTK, DyPE, SEGA (released), SEGA base + temp., Constancy-TM, SEGA + TM.
- English words inside after the change: about 19.
- Caption (`fig:spectrum`): replace the bold first sentence with the bold lead "**Spectra against references and SEGA.**" followed, in regular weight, by "Without SEGA's per-band allocation, luminance power at 8 to 32 pixel wavelengths sits above released SEGA's. Constancy-TM leaves a similar excess, and NTK falls short from 8 to 1024 pixels." Replace "lies above the Nyquist frequency of a 224-pixel view." with "lies above the Nyquist frequency of a 224-pixel view (dotted line). Positive values mean more power than the references (a) or than SEGA (released) (b)." The "195 images" count is already in the caption. The caption then has about 110 tokens with the label.

## A.4 paper equations versus release (page 15, `f6_sega_code.py`)

Current: 5.50 x 5.60 in, the densest figure. Panel a has eight heatmaps with four in-panel range labels ("0.00 to 0.39", ...), two colorbars, and a line plot with a 4-entry legend. Panel b has four photo columns, each a full frame and a crop, above a 4 x 4 table of scores, deltas and intervals (26 printed values). Text: 64 English words, 62 numerals.

What reads as noisy: the score table set inside the figure under the photos, the range labels that repeat the colorbars, and the duplicated axis titles ("Rotary pair, high to low frequency" twice).

Redesign spec:
- Size 5.50 x 4.70 in.
- Panel letters with names (R5): "(a) Gains per pair" and "(b) Samples".
- Panel a:
  - Delete the four range labels; the two colorbars carry the scales.
  - Keep one y-axis title, "Rotary pair", on the left heatmap column only.
  - Replace the line-plot legend with direct labels at the right end, 8 pt: "released" (cobalt) and "paper eqs." (oxblood). The caption explains solid versus dashed.
- Panel b:
  - Delete the score table from the figure and typeset it as a new appendix table (Table D.x, LaTeX, directly under the figure) with the same 16 cells and intervals.
  - Column labels under the crop row (R6), one line each, 8 pt `INK`: "Released, per-step", "Released, static", "Paper eqs., per-step", "Paper eqs., static".
  - Crop location: a red `#FF0000` 1.0 pt box on each full frame and a red 1.0 pt frame on each crop.
  - Delete the rotated "Dev prompt 0, seed 0" label; the caption states it.
- English words inside after the change: about 26.
- Caption (`fig:papercode`): replace the bold first sentence with the bold lead "**Printed equations against the release.**" followed, in regular weight, by "The printed-law implementation produces larger gain changes than the release and degrades this development pilot." Replace "static variants average each method's gains over prompts and steps." with "static variants (dashed) average each method's gains over prompts and steps." Replace the (b) sentence with "(b) Dev prompt 0, seed 0: full frames and a native-resolution crop (red box); Table D.x gives CLIP score and ImageReward means over the eight dev prompts with paired differences from SEGA (released) and 95% bootstrap intervals." The caption then has about 136 tokens with the label.

## F.1 mechanism (page 23, `f7_mechanism.py`)

Current: 5.50 x 4.52 in. Panel a holds a line plot with a secondary σ axis on top, a grey "inactive" region with a two-line sentence, direct labels, and two small heatmaps. Panel b holds three response matrices with a title line and a full-width colorbar (about 0.6 in tall including its label). Text: 40 English words, 43 numerals.

Redesign spec:
- Size 5.50 x 4.00 in.
- Panel letters with names (R5): "(a) Correction size" and "(b) Band response".
- Panel a:
  - Delete the sentence "Late-profile correction is exactly 0 in all 8 prompts for steps 1 to 19 (σ ≥ 0.64)" and keep the grey region; the caption explains it.
  - Keep the top σ axis with 3 ticks (1.0, 0.6, 0.2) at 7 pt.
- Panel b:
  - Delete the title "Width-axis response: 3 prompts, single seed"; it is already in the caption.
  - Matrix titles at 8 pt: "Prompt 0", "Prompt 1", "Prompt 2".
  - Replace the full-width horizontal colorbar with a vertical one, 0.08 in wide, right of the third matrix, with 3 ticks (−1.7, 0, 1.7) at 7 pt and the math label "Δ ln P_out / ln 1.5".
  - Axis ticks: 3 per axis (1/256, 1/32, 1/4) at 7 pt, as now.
- English words inside after the change: about 18.
- Replace the caption in `sec/appendix_impl.tex` (`fig:mechanism`) with this text (about 138 tokens with the label):
  "**Dev-prompt mechanism diagnostics.** The late-profile correction starts late, and the response to a band-gain change differs across prompts. (a) Mean |ln g − ln τ| over spatial rotary pairs per step: median and interquartile band over eight dev prompts, seed 0, with both targets on the NTK base; grey: the correction is exactly zero in all eight prompts for steps 1 to 19 (σ ≥ 0.64). *Right:* median corrections for pairs 0–11; pairs 12–27 are identically zero. (b) Width-axis response matrices for dev prompts 0 to 2, seed 0. Each cell is the same-seed finite-difference change in final-latent log power divided by ln(1.5) when one input octave receives a 1.5× logit gain on both spatial axes throughout sampling. The matrices share one zero-centred colour scale and are not averaged across prompts; no seed-to-seed uncertainty is estimated."
- Acceptance: no sentence in the artwork; caption at most 140 tokens.

## Order of work

1. `crope_style.py`: tick cap and 7 pt ticks, edge marker without value, "(a) Name" panel tags, neutral grid labels under the bottom row, red and cyan crop boxes and frames, prompt bands, metric arrows; the matching `DESIGN_NOTES.md` lines in the same commit.
2. F1 (with the `main.tex` move above the abstract) and F2.
3. F3, then F5, then F4 as a wrapfigure.
4. Appendix figures A.1 to A.4 and F.1.
5. Rebuild, render pages at 160 dpi, recount figure words with the method above and caption tokens with `wc -w` (including the "Figure N:" label), and check each acceptance line.
6. Score the rebuilt figures with the page-by-page rubric of the `gold-standard-paper` skill (part 8) and record the scores in `HANDOVER.md`. Layout note (R10): that rubric asks for at least three wrapped floats in the main text; F4 is the only figure that qualifies, so two main-text tables must also wrap, which this critique does not cover.

## Changes after reference review

- Header: cites LILIE and MFlow as primary references and states the user's standing instruction.
- Word counting: math labels excluded from English-word targets; caption caps counted as whitespace tokens including the label.
- Summary table: targets now F1 18, F4 14, F5 21, A.1 25, A.2 18, A.3 19, A.4 26, F.1 18 (main-text total 93); new caption-cap column; measured columns unchanged.
- Cross-cutting 3: F3 drops the better-side shading as well as the legend; direction moves to arrows in the metric titles.
- Cross-cutting 5: hue cap split into 3 families for diagrams and 4 for plots (R3).
- Cross-cutting 7: bare bold letters become "(a) Name" on F1, F2, A.2, A.3, A.4 and F.1 only (R5).
- Cross-cutting 8 and 9 added: 7 pt type floor; hero placement and crop-box colours.
- Global edits: "(a) Name" tags, grid labels under the bottom row, red and cyan 1.0 pt crop boxes and frames, prompt bands, metric arrows, 7 pt ticks, DESIGN_NOTES update.
- F1: moved above the abstract (`main.tex`); crop boxes red and cyan with matching crop frames instead of white boxes with halos and letters; letter column removed and geometry recomputed (full frame 2.14 in, crops 1.055 in); headers in neutral ink, SEGA (released) bold; panel b names and row labels at 8 pt, ticks at 7 pt, ↑ on metric names; panel names added; caption gains a bold lead, names the crop colours, keeps "seed 0" in (b) and drops the resample count already in the protocol.
- F2: three panels become two ((c) joins (b)); panel names added; a faint real image under the token grid; controller boxes get 0.8 pt borders and 20% tints, arrows 0.6 pt; caption gets a bold lead, a 15-word first sentence and a final pointer to Appendix C and Figures A.4 and F.1, at 110 tokens, within its cap (the current caption has 123).
- F3: better-side shading deleted; ↑ or ↓ added to every metric title; ticks 7 pt; exact caption with a bold lead, the pale-marker definition and the group names.
- F4: becomes a 2.75 x 2.00 in wrapfigure; legend fixed as two rows instead of "legend row or direct labels"; y label in math; annotations at 7 pt; caption gets a bold lead and the dotted-line wording.
- F5: method labels move under the bottom row; prompt excerpts move into grey bands above each prompt instead of the caption; red crop box and crop frame; height 4.05 in; caption gets a bold lead and the shorter row sentence.
- A.1: same as F5 with red and cyan crops; built at 4.85 in and included at 0.88 linewidth so the added prompt bands still fit one page; letter column removed; exact caption edits.
- A.2: panel names; row labels rotated in neutral ink; column labels under the images; red and cyan windows; period labels shortened; exact caption edits with the ratio-to-arm check.
- A.3: full width instead of 0.88; panel names; math-only y labels; one 8 pt legend row; caption bold lead split from the 31-word claim.
- A.4: panel names; column labels under the crop row; red crop box; exact caption edits including "(dashed)" and the Table D.x pointer.
- F.1: panel names; ticks at 7 pt; full caption replacement with a bold lead, kept under the 140-token appendix cap.
- Order of work: adds the style-module items, the hero move, the token check and the rubric scoring.
