# Figure captions (figv3 2026-10-19; v4 figure update 2026-10-24)

Final caption text for every figure. The compiled captions live in the files named below; copy each
caption there verbatim (macros as in the current paper). Each starts with a bold lead of 3 to 6 words
and walks multi-panel figures by letter. Spelling follows American usage (gray, color, center).

v4 (gate review v3, `.agent/notes/fable_review_v3_2026-10-24.md`, Sections 4 and 5) changed Figures 1 and 2,
replaced the body Figure 4 with a half-width two-panel version and added a body qualitative grid (Figure 5)
and a full-width appendix version of the confirmation figure. Blocks of figures v4 did not touch keep their
figv3 text under their current paper numbers; for those, the captions in `sec/*.tex` are authoritative.

Naming note: `f4_spectrum.pdf` and `fA_spectrum_full.pdf` carry the registered-confirmation figure
(`fig:fs-confirm`, formerly `figs/strengthen/fs_confirm.pdf`), not the luminance spectrum. The file names were
set by the integration plan. The spectrum in the body is still Figure 3, `f8_spectrum_main.pdf`, unchanged.

| Paper (2026-10-24) | Asset | Caption source | Label | Include | Size (in) | English words inside |
| :-- | :-- | :-- | :-- | :-- | :-- | --: |
| Figure 1 | f1_hero.pdf (v4) | sec/hero.tex | fig:hero | `\linewidth` | 5.50 x 3.12 | 18 |
| Figure 2 | f2_method.pdf (v4) | sec/background.tex | fig:method | `\linewidth` | 5.50 x 2.17 | 19 (plus "ln") |
| Figure 3 | f8_spectrum_main.pdf | sec/results_decomp.tex | fig:spectrum_main | wrapfigure 0.5, include `\linewidth` | 2.75 x 2.00 | 11 |
| Figure 4 | f4_spectrum.pdf (v4, new) | sec/results_confirmation.tex | fig:fs-confirm | wrapfigure 0.5, include `\linewidth` | 2.75 x 1.83 | 13 |
| Figure 5 | f5_qualitative_body.pdf (v4, new) | Section 5.3 (new float) | fig:qual_body (suggested) | `\linewidth` | 5.50 x 3.40 | 14 |
| Figure A.1 | f3_forest.pdf | sec/appendix_figs.tex | fig:forest | `\linewidth` | 5.50 x 2.90 | 29 |
| Figure A.2 | f4_qualitative_main.pdf | sec/appendix_figs.tex | fig:qual_main | `\linewidth` | 5.50 x 4.30 | 16 |
| Figure A.3 | f8_spectrum.pdf | sec/appendix_figs.tex | fig:spectrum | `\linewidth` | 5.50 x 2.45 | 20 |
| Figure A.4 | f4_qualitative.pdf | sec/appendix_figs.tex | fig:qual | `\linewidth` | 5.50 x 3.10 | 16 |
| Figure A.5 | f6_sega_code.pdf | sec/appendix_figs.tex | fig:papercode | `\linewidth` | 5.50 x 4.73 | 33 |
| Figure A.6 | f5_checkpoint.pdf | sec/appendix_figs.tex | fig:checkpoint | `\linewidth` | 5.50 x 3.26 | 16 |
| Appendix C (new) | fA_spectrum_full.pdf (v4, new) | tables/float_strengthen_appendix.tex or app:confirm | fig:fs-confirm-full (suggested) | `\linewidth` | 5.50 x 1.62 | 19 |
| Figure F.1 | fA_method_zoom.pdf | sec/appendix_impl.tex | fig:method_zoom | `\linewidth` | 5.50 x 3.53 | 30 (plus equations) |
| Figure G.1 | f7_mechanism.pdf | sec/appendix_impl.tex | fig:mechanism | `\linewidth` | 5.50 x 4.02 | 22 |

## Figure 1: f1_hero.pdf (`fig:hero`, v4)

Change: a solid black line across each metric's three bars marks the total gain, SEGA (released) minus NTK,
named "Total" once (ImageReward panel). The bars sum to it exactly. It meets the temperature bar on both
reward models and sits above it on full-frame MUSIQ by the allocation's share.

```latex
\textbf{Temperature does the work; allocation adds full-frame MUSIQ.} (a)~On \N{st@f@n} held-out prompts (seed 0), bars split the gain of \segaofficial{} over \ntk{} (dashed line at zero, solid line at the total, which the three bars sum to) into the temperature and base main effects of the 2$\times$2 factorial and the allocation (\segaofficial{} minus \tempsb); the interaction survives Holm on no metric and is not drawn. Bars are paired mean differences with nominal 95\% prompt-bootstrap intervals (2{,}000 draws); $\star$ marks a Holm-adjusted $p\le0.05$ over the 80 registered factorial tests. (b)~Illustrative held-out row 32 was chosen by its cabin caption before viewing; native $384^2$ crops (cabin red, mountain cyan) at $(2048,3200)$ and $(2048,1536)$ were fixed after viewing and are identical for every arm.
```

## Figure 2: f2_method.pdf (`fig:method`, v4)

Change: a three-glyph legend under the ln g panel (circle SEGA, diamond Constancy-TM, outlined bar for the
temperature); the panel is 1.36 in wide (was 1.00), bars are outlined, marks larger, math subscripts 6.5 pt.

```latex
\textbf{Three knobs act on the logit.} (a)~Four of the \N{c@pairsperaxis} width-axis bands of \cref{eq:fourier}, drawn as unit-amplitude, zero-phase cosines under native base $\theta$ (dashed) and released base $\theta'$, are read at the key-query offset $\Delta$. Log gains $\ln g_{a,j}$ at step 14 of the first held-out prompt (row 8, seed 0), whose \segaofficial{} output is shown, are marked as in the legend: \segaofficial{} circles, \constancytm{} diamonds and \tempsb{} bars, which end at $\ln\tau$ (dashed) on every band; the solid gray line is zero. (b)~Either controller reads $x_t$ and sets $g_{a,j}$; \cref{fig:method_zoom} draws both at component level.
```

## Figure 4: f4_spectrum.pdf (`fig:fs-confirm`, v4, replaces figs/strengthen/fs_confirm.pdf in the body)

Two reward panels at half text width; both show SEGA base + temp. minus SEGA (released), written once as a
header over the panels, so the body figure has no sign flip. The MUSIQ panel moves to the appendix version
below; the text keeps the H2 number. The caption is kept to seven lines at half width (measured in a test
build with the paper's geometry and caption style); full sentences ran to ten lines and three more wrap lines.

```latex
\textbf{Equivalence holds on pooled prompts.} \tempsb{} minus \segaofficial. Filled: registered H1 on \N{st@h@n@ir} pooled prompts, inside the amended margin (gray). Hollow: \N{st@e@n} fresh (PickScore leaves the margin), \N{st@f@n} held-out at seed 0, and held-out means over seeds 0 to 2. Nominal 95\% prompt-bootstrap intervals; \cref{fig:fs-confirm-full} adds MUSIQ.
```

## Figure 5: f5_qualitative_body.pdf (`fig:qual_body`, v4, new body float in Section 5.3)

Selection: the first two held-out manifest rows (8 and 9), fixed before any output was viewed; seed 0, one
sample per arm, n = 2 prompts, descriptive only (no statistic). Crops: 384 px native windows at the shared
location of Figure A.2 (the script asserts identical boxes), Flate-lossless and byte-checked
(`f5_qualitative_body_audit.json`). What is visible at native scale, checked at 1:1 on the source PNGs: NTK
smears texture (and its row 9 window shows the river, not the crowd); DyPE adds colored specks; SEGA base +
temp. and Constancy-TM show bright or colored specks in foliage, sky and pavement; SEGA (released) and
SEGA + TM are clean. The specks are a few pixels wide, inside the 8 to 32 px band of Figure 3. The caption
sets four lines at full width; figure plus caption measure 4.04 in, 0.449 of the 9 in text block. Figure A.2's
caption states the crop-location rule (output-dependent, review stage); the body caption points to it.

```latex
\textbf{Speckle where the allocation is absent.} Held-out rows 8 and 9, the first two held-out prompts, were fixed before viewing (seed 0); native 384-pixel crops sit below Lanczos-reduced frames at the locations of \cref{fig:qual_main}. \tempsb{} and \constancytm{}, which lack \sega's allocation, show specks that \segaofficial{} and \segactm{} lack; \dype{} adds colored specks and \ntk{} loses detail.
```

## Appendix: fA_spectrum_full.pdf (`fig:fs-confirm-full`, v4, new)

The full-width confirmation figure with all three panels (the content of the former body Figure 4). Each sign
is a header over the panels it governs, with a thin rule spanning them, so the flip at MUSIQ is explicit.

```latex
\textbf{Registered confirmation, all three panels.} ImageReward and PickScore show \tempsb{} minus \segaofficial{} with the amended equivalence margins (gray band, dotted edges); full-frame MUSIQ shows the opposite difference, \segaofficial{} minus \tempsb, the effect of the allocation. Filled marks are the registered tests (H1 on the \N{st@h@n@ir} pooled prompts, H2 on the \N{st@e@n} fresh prompts); hollow marks are the fresh prompts alone, the \N{st@f@n} held-out prompts at seed 0 and their means over seeds 0 to 2. Whiskers are nominal 95\% paired prompt-bootstrap intervals (2{,}000 draws); \cref{tab:strengthen-fresh-tests} lists the values and Holm-adjusted $p$.
```

## Figure A.1: f3_forest.pdf (`fig:forest`)

```latex
\textbf{Paired differences from \segaofficial.} No per-band law improves on \segaofficial{} in any reward or image-quality metric, and \dype's isolated gains in crop MUSIQ and CLIP-IQA come with losses elsewhere. The dashed blue line is \segaofficial; marks show held-out paired differences with 95\% bootstrap intervals (rows 8--194, seed 0, $n=\N{ho@nprompts}$), pale where the interval includes 0, and \cref{tab:main} lists them. Hollow marks: seed-1 replication on rows 8--62 ($n=\N{so@n@tsb}$) for \tempsb{} and \constancytm{} (none for \segactm); triangles: \ntk{} beyond the axis limit; arrows: better direction.
```

## Figure 3: f8_spectrum_main.pdf (`fig:spectrum_main`)

```latex
\textbf{Paired luminance spectra.} Removing the allocation (\tempsb) leaves more octave-band luminance power at 8 to 32 pixel wavelengths (gray band) than \segaofficial, the dashed blue zero line. The band lies above the Nyquist frequency of a 224-pixel view (dashed vertical); intervals are 95\% prompt-bootstrap over held-out rows 8--194 ($n=\N{ho@nprompts}$, seed 0). This contrast accompanies the full-frame MUSIQ gap but does not isolate its cause; \cref{fig:spectrum} in \cref{app:figs} adds errors against the reference profile.
```

## Figure A.2: f4_qualitative_main.pdf (`fig:qual_main`)

```latex
\textbf{Native crops show what thumbnails hide.} Held-out rows 8, 9 and 10 (left to right), the first three held-out prompts, were fixed before any output was viewed; seed 0, each Lanczos-reduced full frame beside its unaltered 384-pixel crop (red box). During review, we chose one shared crop location per row on a fixed grid by maximizing the minimum Laplacian energy across all six arms, each normalized by its 90th percentile over candidate locations. This output-dependent rule favors detail present in every arm and selects no winning method; differing compositions remain a limitation. Full prompts are in \cref{fig:qual}.
```

## Figure A.4: f4_qualitative.pdf (`fig:qual`)

```latex
\textbf{\ntk{} loses fine detail at $4096^2$.} The other arms render comparable detail, and \dype{} shows faint speckle in flat sky on held-out row 8 (one crop checked); compositions differ between arms, so a crop can show different content. Held-out rows 8, 9 and 10 (left to right), the first three held-out rows, were fixed before any output was viewed; seed 0, one sample per arm. Full frames are Lanczos-reduced; native-resolution crops sit at identical coordinates and are never resampled (red: highest-detail tile of the across-arm mean luminance; cyan: image center). Prompts, left to right: ``A stone bridge with arched spans reflects golden light on the water, flanked by buildings and a leafy tree under a clear blue sky.'' ``A panoramic view of Paris featuring the Eiffel Tower, the Seine River, and various ornate buildings with greenery and bustling activity, including boats on the water and crowds along walkways.'' ``A cluttered artist's corner features an ornate gold frame, a painting of a man with glasses, several empty picture frames, a vintage suitcase, and various brushes in a tin can, all set against a textured gray wall and draped with patterned fabric.''
```

## Figure A.6: f5_checkpoint.pdf (`fig:checkpoint`)

```latex
\textbf{Lattice and checkpoint check.} The development comparison shows a lattice on \fluxdev{}, but checkpoint and guidance both change. (a)~Dev prompt 0 under official \dype{} code (seed 42) and our \dype{} and \segaofficial{} pipelines (seed 0), at guidance 3.5 (\fluxdev) and 4.5 (\krea). Red and cyan boxes mark native 192-pixel windows at the same place in every image. (b)~Whitened radial power spectra of luminance: \fluxdev{} (black) peaks at the lattice period (dot) and \krea{} (gray) does not in these samples. Peak ratios: 36$\times$ for the official-code pair; over eight prompts in our pipeline, 12$\times$ [7, 20] for \dype{} and 5$\times$ [2, 11] for \segaofficial{} (geometric means, 95\% bootstrap intervals). Official-code evidence covers one prompt per checkpoint, and the figure does not isolate a checkpoint effect.
```

## Figure A.3: f8_spectrum.pdf (`fig:spectrum`)

```latex
\textbf{Spectra against references and \sega.} Without \sega's per-band allocation, luminance power at 8 to 32 pixel wavelengths (gray band) sits above released \sega's; \constancytm{} leaves a similar excess, and \ntk{} falls short from 8 to 1024 pixels. (a)~Log-power error against the mean $\ln P$ of the 195 \afk{} references, per octave band of linear-sRGB luminance; (b)~paired difference from \segaofficial, the dashed blue zero line; positive means more power. Held-out rows 8--194 ($n=\N{ho@nprompts}$, seed 0) with 95\% bootstrap intervals over prompts; the dashed vertical marks the Nyquist frequency of a 224-pixel view.
```

## Figure A.5: f6_sega_code.pdf (`fig:papercode`)

```latex
\textbf{Printed equations against the release.} The printed-law implementation produces larger gain changes than the release and degrades this development pilot. Eight dev prompts (rows 0--7, seed 0), descriptive only. (a)~Median log logit gain per rotary pair (pair 0 has the highest frequency) and denoising step. Released and printed-law heatmaps have separate symmetric color scales, so equal colors across the groups do not mean equal magnitudes. The line panel uses one shared axis for mean log gain over prompts, steps and both spatial axes; static variants (dashed) average each method's gains over prompts and steps. (b)~Dev prompt 0, seed 0: full frames and a native-resolution crop (red box). \Cref{tab:papercode_scores} gives CLIP score and ImageReward means over the eight dev prompts, with paired differences from \segaofficial{} and 95\% bootstrap intervals.
```

New table directly under Figure A.4 (`figs/f6_sega_code_table.tex` is written by `f6_sega_code.py`):

```latex
\begin{table}[H]
\centering
\caption{\textbf{Scores of the paper-equation pilot.} CLIP score and ImageReward means over the eight dev prompts (rows 0--7, seed 0) for the four variants of \cref{fig:papercode}, with the paired difference from \segaofficial{} and its 95\% percentile bootstrap interval (2,000 resamples over prompts); descriptive only. The \segaofficial{} row is shaded. Key result: our implementation of the printed equations scores lower than the release on both metrics, and its static variant lowers CLIP score to 13.06.}
\label{tab:papercode_scores}
\footnotesize
\input{figs/f6_sega_code_table}
\end{table}
```

## Figure F.1: fA_method_zoom.pdf (`fig:method_zoom`)

```latex
\textbf{Both gain controllers at component level.} (a)~The released \sega{} sensor of \cref{fig:method}b reads the channel mean of the projected input features, maps the axis profile to rotary pairs by rank and scales its tilt by the radial flatness. (b)~\constancytm{} gates a bounded log-power correction toward the time-matched native target and adds it to the \sega{} temperature. White boxes are inputs and outputs; the equations repeat \cref{alg:sega,alg:ctm} and \cref{app:impl}.
```

## Figure G.1: f7_mechanism.pdf (`fig:mechanism`)

```latex
\textbf{Dev-prompt mechanism diagnostics.} The late-profile correction starts late, and the response to a band-gain change differs across prompts. (a)~Mean $|\ln g-\ln\tau|$ over spatial rotary pairs per step: median and interquartile band over eight dev prompts, seed 0, with both targets on the \ntk{} base. In the gray region the late-profile correction is exactly zero in all eight prompts (steps 1 to 19, $\sigma\geq 0.64$). \emph{Right:} median corrections for pairs 0--11; pairs 12--27 are identically zero. (b)~Width-axis response matrices for dev prompts 0 to 2, seed 0. Each cell is the same-seed finite-difference change in final-latent log power divided by $\ln(\NmechProbeFactor)$ when one input octave receives a $\NmechProbeFactor\times$ logit gain on both spatial axes throughout sampling. The matrices share one zero-centerd color scale and are not averaged across prompts; no seed-to-seed uncertainty is estimated.
```


## LaTeX handoff for the text agent

### v4 (2026-10-24)

- **Figure 1** (`sec/hero.tex`): same size (5.50 x 3.12 in); new caption above (adds the solid total line).
- **Figure 2** (`sec/background.tex`): same size (5.50 x 2.17 in); new caption above. Rebuild order unchanged:
  `f2_method_data.py` (only if the data change), `tectonic -X compile f2_method.tex`, then `f2_method_export.py`.
- **Figure 4** (`sec/results_confirmation.tex`, keep `\label{fig:fs-confirm}`): replace the full-width float with
  a right wrapfigure:
  `\begin{wrapfigure}[18]{r}{0.5\linewidth}\vspace{-12pt}\centering\includegraphics[width=\linewidth]{figs/f4_spectrum.pdf}`
  then caption, label and `\vspace{-10pt}\end{wrapfigure}`. The figure is 2.75 x 1.83 in and prints at scale 1.0
  (tick labels 7 pt) only at `width=\linewidth` inside the wrap. Line count: about 11 lines for the figure plus
  one per caption line (the caption above sets about seven, so `[18]`); check that no narrow orphan lines remain.
  It needs about 18 wrapped lines away from section starts, lists and page breaks.
- **Figure 5** (new, Section 5.3, suggested `\label{fig:qual_body}`): `\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figs/f5_qualitative_body.pdf}` with the caption above. Figure 3.40 in tall
  (0.38 of the text block); with the four-line caption above the float measures 4.04 in (0.449). The tile side
  `T = 0.74` in `f5_qualitative_body.py` sets the height (height = 4 T + 0.44 in).
- **Appendix** (new float near `tab:strengthen-fresh-tests` in `app:confirm`, suggested
  `\label{fig:fs-confirm-full}`): `\includegraphics[width=\linewidth]{figs/fA_spectrum_full.pdf}`, 5.50 x 1.62 in.
  The Figure 4 caption cites it; if it is not added, drop that clause.
- `figs/strengthen/fs_confirm.pdf` and its script stay untouched; `f4_spectrum.py` reads the same analysis files
  through `fs_confirm.family` and checks every registered number against `tables/num_strengthen.tex`.

### figv3 (2026-10-19)

Every figure is built at its final print size; the include widths in the table above give scale 1.0.

- **F1** (`sec/hero.tex`): new caption; placeholder height 3.12in. The chart is now panel (a), so in
  `sec/results_decomp.tex` change `\cref{fig:hero}b` to `\cref{fig:hero}a`. Page 1 fits with the hero under
  the abstract (0.14 in to spare); moving it above the abstract needs no figure change.
- **F2** (`sec/background.tex`, `fig:method`): new caption; placeholder height 2.17in (was 2.37in). Its last sentence cites `\cref{fig:method_zoom}`, which needs the new appendix float below. Rebuild order: `f2_method_data.py`, tectonic on `f2_method.tex`, then `f2_method_export.py`.
- **F3** (`sec/results_decomp.tex`): new caption; placeholder height 2.9in.
- **F4** (`fig:spectrum_main`): the figure is 2.75 in wide for a right wrapfigure:
  `\begin{wrapfigure}[26]{r}{0.5\linewidth} \centering \figorplaceholder[\linewidth]{figs/f8_spectrum_main.pdf}{2.0in}`
  then caption and label. Inside a wrapfigure `\linewidth` is already the wrap width, so `[0.5\linewidth]` there
  would print it at 1.375 in. It needs about 25 wrapped lines away from section starts, lists and page breaks.
- **F5** (`fig:qual_main`): new caption (full prompts moved to Figure A.1); placeholder height 4.30in. The
  sentence "Compositions differ in \cref{fig:qual_main}, so its crops are illustrative; \cref{fig:qual} retains
  the original three-prompt grid." is out of date: both figures now show rows 8 to 10, and A.1 adds a second crop
  at the image center.
- **A.1** (`sec/appendix_figs.tex`, `fig:qual`): include `\linewidth` (was 0.95), placeholder 3.10in; the caption
  now carries the three full prompts.
- **A.2** (`fig:checkpoint`): new caption only.
- **A.3** (`fig:spectrum`): include `\linewidth` (was 0.88), placeholder 2.45in.
- **A.4** (`fig:papercode`): new caption; add the score table float above directly after the figure with
  `\label{tab:papercode_scores}` (`tab:papercode` is taken). `figs/f6_sega_code_table.tex` is tabular-only and
  calls `\providecolor{cMethodShade}{HTML}{E6F0F6}`; placeholder 4.73in.
- **Appendix zoom** (`fig:method_zoom`): new float in `sec/appendix_impl.tex` directly before `\subsection{Per-band control laws}\label{app:laws}`: `\begin{figure}[!htbp] \centering \figorplaceholder[\linewidth]{figs/fA_method_zoom.pdf}{3.53in}` with the caption above and `\label{fig:method_zoom}`. Suggested pointer in the released-SEGA paragraph: "\Cref{fig:method_zoom} draws both controllers at component level." The figure writes the clean estimate as $\hat x_0$ (as Algorithm 2 does); `app:laws` writes $\hat x$, so pick one form.
- **F.1** (`sec/appendix_impl.tex`, `fig:mechanism`): new caption; placeholder 4.02in.
- **Palette in the preamble**: `papertint` already equals the method shade `#E6F0F6`. The algorithm colors in
  `style.tex` still use the old palette (`codekw` 1F4E8C, `codefn` A2467F, `codecm` 2E7D5B); the paper palette
  equivalents are keywords `0068A8`, functions `C878A0`, comments `009870` (the LILIE scheme).
- **Spelling**: the captions use gray, color and center; the text mixes in "colour" (2) and "centre" (1).
