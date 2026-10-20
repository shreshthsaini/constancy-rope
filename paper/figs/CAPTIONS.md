# Figure captions (figv3, 2026-10-19)

Final caption text for every figure. The compiled captions live in the files named below; copy each
caption there verbatim (macros as in the current paper). Each starts with a bold lead of 3 to 6 words
and walks multi-panel figures by letter. Spelling follows American usage (gray, color, center).

| Figure | Asset | Caption source | Label | Include | Placeholder height | English words inside |
| :-- | :-- | :-- | :-- | :-- | :-- | --: |
| F1 | f1_hero.pdf | sec/hero.tex | fig:hero | `\linewidth` | 3.12in | 17 |
| F2 | f2_method.pdf | sec/methods.tex | fig:method | `\linewidth` | 2.17in | 17 |
| F3 | f3_forest.pdf | sec/results_decomp.tex | fig:forest | `\linewidth` | 2.90in | 29 |
| F4 | f8_spectrum_main.pdf | sec/results_decomp.tex | fig:spectrum_main | `wrapfigure{r}{0.5\linewidth}, include [\linewidth]` | 2.0in | 11 |
| F5 | f4_qualitative_main.pdf | sec/results_decomp.tex | fig:qual_main | `\linewidth` | 4.30in | 16 |
| A1 | f4_qualitative.pdf | sec/appendix_figs.tex | fig:qual | `\linewidth (was 0.95)` | 3.10in | 16 |
| A2 | f5_checkpoint.pdf | sec/appendix_figs.tex | fig:checkpoint | `\linewidth` | 3.26in | 16 |
| A3 | f8_spectrum.pdf | sec/appendix_figs.tex | fig:spectrum | `\linewidth (was 0.88)` | 2.45in | 20 |
| A4 | f6_sega_code.pdf | sec/appendix_figs.tex | fig:papercode | `\linewidth` | 4.73in | 33 |
| FA | fA_method_zoom.pdf | new, appendix_impl.tex beside the controller equations | fig:method_zoom | `\linewidth` | 3.53in | 30 (plus equations) |
| F1mech | f7_mechanism.pdf | sec/appendix_impl.tex | fig:mechanism | `\linewidth` | 4.02in | 22 |

## F1: f1_hero.pdf (`fig:hero`)

```latex
\textbf{Where \sega's gain comes from.} (a)~\tempsb{} minus \ntk{} (base\,+\,temp.) carries most of the gain; \segaofficial{} minus \tempsb{} (allocation) adds full-frame MUSIQ. Bars and whiskers are means and nominal 95\% paired bootstrap intervals over \N{ho@nprompts} held-out prompts, seed 0, one scale per metric. (b)~Illustrative held-out row 32 was chosen by its cabin caption before viewing; native $384^2$ crops (cabin red, mountain cyan) were fixed after viewing at $(2048,3200)$ and $(2048,1536)$, identical for every arm.
```

## F2: f2_method.pdf (`fig:method`)

```latex
\textbf{Three knobs act on the logit.} (a)~Four of the \N{c@pairsperaxis} width-axis bands of \cref{eq:fourier}, drawn as unit-amplitude, zero-phase cosines under native base $\theta$ (dashed) and released base $\theta'$, are read at the key-query offset $\Delta$. Log gains at step 14 of the first held-out prompt, whose \segaofficial{} output is shown, are those of \tempsb{} (bars), \segaofficial{} (circles) and \constancytm{} (diamonds). (b)~Either controller reads $x_t$ and sets $g_{a,j}$; \cref{fig:method_zoom} draws both at component level.
```

## F3: f3_forest.pdf (`fig:forest`)

```latex
\textbf{Paired differences from \segaofficial.} No per-band law improves on \segaofficial{} in any reward or image-quality metric, and \dype's isolated gains in crop MUSIQ and CLIP-IQA come with losses elsewhere. The dashed blue line is \segaofficial; marks show held-out paired differences with 95\% bootstrap intervals (rows 8--194, seed 0, $n=\N{ho@nprompts}$), pale where the interval includes 0, and \cref{tab:main} lists them. Hollow marks: seed-1 replication on rows 8--62 ($n=\N{so@n@tsb}$) for \tempsb{} and \constancytm{} (none for \segactm); triangles: \ntk{} beyond the axis limit; arrows: better direction.
```

## F4: f8_spectrum_main.pdf (`fig:spectrum_main`)

```latex
\textbf{Paired luminance spectra.} Removing the allocation (\tempsb) leaves more octave-band luminance power at 8 to 32 pixel wavelengths (gray band) than \segaofficial, the dashed blue zero line. The band lies above the Nyquist frequency of a 224-pixel view (dashed vertical); intervals are 95\% prompt-bootstrap over held-out rows 8--194 ($n=\N{ho@nprompts}$, seed 0). This contrast accompanies the full-frame MUSIQ gap but does not isolate its cause; \cref{fig:spectrum} in \cref{app:figs} adds errors against the reference profile.
```

## F5: f4_qualitative_main.pdf (`fig:qual_main`)

```latex
\textbf{Native crops show what thumbnails hide.} Held-out rows 8, 9 and 10 (left to right), the first three held-out prompts, were fixed before any output was viewed; seed 0, each Lanczos-reduced full frame beside its unaltered 384-pixel crop (red box). During review, we chose one shared crop location per row on a fixed grid by maximizing the minimum Laplacian energy across all six arms, each normalized by its 90th percentile over candidate locations. This output-dependent rule favors detail present in every arm and selects no winning method; differing compositions remain a limitation. Full prompts are in \cref{fig:qual}.
```

## A1: f4_qualitative.pdf (`fig:qual`)

```latex
\textbf{\ntk{} loses fine detail at $4096^2$.} The other arms render comparable detail, and \dype{} shows faint speckle in flat sky on held-out row 8 (one crop checked); compositions differ between arms, so a crop can show different content. Held-out rows 8, 9 and 10 (left to right), the first three held-out rows, were fixed before any output was viewed; seed 0, one sample per arm. Full frames are Lanczos-reduced; native-resolution crops sit at identical coordinates and are never resampled (red: highest-detail tile of the across-arm mean luminance; cyan: image center). Prompts, left to right: ``A stone bridge with arched spans reflects golden light on the water, flanked by buildings and a leafy tree under a clear blue sky.'' ``A panoramic view of Paris featuring the Eiffel Tower, the Seine River, and various ornate buildings with greenery and bustling activity, including boats on the water and crowds along walkways.'' ``A cluttered artist's corner features an ornate gold frame, a painting of a man with glasses, several empty picture frames, a vintage suitcase, and various brushes in a tin can, all set against a textured gray wall and draped with patterned fabric.''
```

## A2: f5_checkpoint.pdf (`fig:checkpoint`)

```latex
\textbf{Lattice and checkpoint check.} The development comparison shows a lattice on \fluxdev{}, but checkpoint and guidance both change. (a)~Dev prompt 0 under official \dype{} code (seed 42) and our \dype{} and \segaofficial{} pipelines (seed 0), at guidance 3.5 (\fluxdev) and 4.5 (\krea). Red and cyan boxes mark native 192-pixel windows at the same place in every image. (b)~Whitened radial power spectra of luminance: \fluxdev{} (black) peaks at the lattice period (dot) and \krea{} (gray) does not in these samples. Peak ratios: 36$\times$ for the official-code pair; over eight prompts in our pipeline, 12$\times$ [7, 20] for \dype{} and 5$\times$ [2, 11] for \segaofficial{} (geometric means, 95\% bootstrap intervals). Official-code evidence covers one prompt per checkpoint, and the figure does not isolate a checkpoint effect.
```

## A3: f8_spectrum.pdf (`fig:spectrum`)

```latex
\textbf{Spectra against references and \sega.} Without \sega's per-band allocation, luminance power at 8 to 32 pixel wavelengths (gray band) sits above released \sega's; \constancytm{} leaves a similar excess, and \ntk{} falls short from 8 to 1024 pixels. (a)~Log-power error against the mean $\ln P$ of the 195 \afk{} references, per octave band of linear-sRGB luminance; (b)~paired difference from \segaofficial, the dashed blue zero line; positive means more power. Held-out rows 8--194 ($n=\N{ho@nprompts}$, seed 0) with 95\% bootstrap intervals over prompts; the dashed vertical marks the Nyquist frequency of a 224-pixel view.
```

## A4: f6_sega_code.pdf (`fig:papercode`)

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

## FA: fA_method_zoom.pdf (`fig:method_zoom`)

```latex
\textbf{Both gain controllers at component level.} (a)~The released \sega{} sensor of \cref{fig:method}b reads the channel mean of the projected input features, maps the axis profile to rotary pairs by rank and scales its tilt by the radial flatness. (b)~\constancytm{} gates a bounded log-power correction toward the time-matched native target and adds it to the \sega{} temperature. White boxes are inputs and outputs; the equations repeat \cref{alg:sega,alg:ctm} and \cref{app:impl}.
```

## F1mech: f7_mechanism.pdf (`fig:mechanism`)

```latex
\textbf{Dev-prompt mechanism diagnostics.} The late-profile correction starts late, and the response to a band-gain change differs across prompts. (a)~Mean $|\ln g-\ln\tau|$ over spatial rotary pairs per step: median and interquartile band over eight dev prompts, seed 0, with both targets on the \ntk{} base. In the gray region the late-profile correction is exactly zero in all eight prompts (steps 1 to 19, $\sigma\geq 0.64$). \emph{Right:} median corrections for pairs 0--11; pairs 12--27 are identically zero. (b)~Width-axis response matrices for dev prompts 0 to 2, seed 0. Each cell is the same-seed finite-difference change in final-latent log power divided by $\ln(\NmechProbeFactor)$ when one input octave receives a $\NmechProbeFactor\times$ logit gain on both spatial axes throughout sampling. The matrices share one zero-centerd color scale and are not averaged across prompts; no seed-to-seed uncertainty is estimated.
```


## LaTeX handoff for the text agent

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
