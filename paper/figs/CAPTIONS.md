# Figure captions (finding first)

Paste-ready LaTeX. Every number below is printed by, or recomputable from, the script beside the figure.
Shared facts: FLUX.1-Krea-dev, 4096 x 4096, 28 steps, guidance 4.5, held-out rows 8 to 194 (n = 187), seed 0,
paired with released SEGA on the same prompt and seed; 95% CIs are percentile intervals of 2,000 paired
bootstrap resamples over prompts (seed 0), not adjusted for multiple comparisons.

## F1 `f1_hero.pdf`
\caption{\textbf{Released SEGA's 4K gain over the NTK base comes mostly from its RoPE base and one global
attention temperature; the per-band allocation adds little.}
(a) Held-out row 28 (chosen by subject before any output was viewed), seed 0, $4096^2$. Left: released SEGA's
full frame, Lanczos-reduced. Right: native 384-pixel crops, never resampled, at identical coordinates in every
arm (A: highest-detail tile of the across-arm mean luminance; B: image centre). Columns add one ingredient at a
time: NTK (RoPE base only), SEGA base + temp. (SEGA's released base and global temperature), SEGA (released)
(plus the per-band allocation). Compositions differ between arms, so a crop can show different content.
(b) Share of released SEGA's gain over NTK recovered by SEGA base + temp.,
$(\bar{x}_{\text{base+temp.}}-\bar{x}_{\text{NTK}})/(\bar{x}_{\text{SEGA}}-\bar{x}_{\text{NTK}})$, on held-out
rows 8 to 194 ($n = 187$, seed 0); bracketed values and whiskers are 95\% percentile intervals from 2,000
joint bootstrap resamples of prompts, ratio recomputed per resample. The dark segment is the remainder up to
released SEGA, the part the per-band allocation adds.}

## F2 `f2_method.pdf`
\caption{\textbf{The RoPE base and one global temperature are the big levers on the attention logit; per-band
gains only fine-tune the band weights.} (a) Query $p$, key $p'$ and offset $\Delta$ on the $256^2$ token grid
of a $4096^2$ image; the inset shows band 8 under the native base $b$ (dashed) and SEGA's released base $b'$.
(b) Four of the 28 width-axis bands at their true wavelengths with their log gains (sky: $\ln\tau$, equal for
every band; circles: released SEGA; diamonds: Constancy-TM), taken from held-out row 8, seed 0, step 14 of 28.
The curves on the right sum only the four drawn bands (unit content), under $b$, $b'$, and $b'$ with $\tau$.
(c) Either controller reads the latent $x_t$ at every step and sets $g_j$: SEGA from z-scored band energy around
a reference gain, Constancy-TM through an efficient-coding gate and a time-matched native target.}

## F3 `f3_forest.pdf`
\caption{\textbf{Removing SEGA's per-band allocation leaves every reward-metric difference with a 95\% CI that
includes zero but costs about 2 full-frame MUSIQ points; no principled per-band rule improves on released SEGA
in any reward or full-frame quality metric.} Paired differences, method minus released SEGA, on held-out rows
8 to 194 ($n = 187$, seed 0). Hollow markers: seed-1 replication on rows 8 to 62
($n = 55$) for SEGA base + temp. and Constancy-TM. Pale marks: the CI includes zero; the shaded half of each
panel is the better direction. NTK values beyond the axis are printed at its edge. Isolated per-metric gains
(DyPE on crop MUSIQ and CLIP-IQA, lower tiling for SEGA base + temp. and Constancy-TM)
come with losses elsewhere. A CI that includes zero does not establish equivalence.}

## F4 `f4_qualitative.pdf`
\caption{\textbf{At 4096 pixels NTK loses fine detail, while released SEGA, SEGA base + temp. and the
per-band variants render comparable detail.} Held-out rows 8 to 10, the first three held-out rows, fixed
before any output was viewed; seed 0, one sample per method. Full frames are Lanczos-reduced; native 384-pixel
crops sit at identical coordinates (A: highest-detail tile of the across-method mean luminance; B: image
center), never resampled. Compositions differ between methods, so a crop can show different content (row 8,
B). DyPE shows faint speckle in flat sky (row 8, A).}

## F5 `f5_checkpoint.pdf`
\caption{\textbf{FLUX.1-dev breaks into a dot lattice at $4096^2$ even under the official DyPE code, while
FLUX.1-Krea-dev shows no lattice peak; results on one checkpoint do not transfer.} (a) Dev prompt 0 (the first
manifest row). Official DyPE code at its seed 42; our pipeline at seed 0; guidance 3.5 (dev) and 4.5 (Krea).
Crops A and B are native 192-pixel windows centred at (2048, 1024) and (2048, 3072). (b) Whitened radial power
spectra of luminance; dev spectra peak at a period of 25 to 29 pixels. Dev-to-Krea peak ratio: 36$\times$ for
the single official pair; over the 8 dev prompts (geometric mean, 95\% bootstrap CI, 2,000 resamples) 12$\times$
[7, 20] for DyPE and 5$\times$ [2, 11] for released SEGA.}

## F6 `f6_sega_code.pdf`
\caption{\textbf{SEGA as written in the paper is not SEGA as released: the paper equations swing the log logit
gain from $-0.79$ to $+1.74$ (released code: 0.00 to 0.39), degrading images, and their static tilt destroys
them.} (a) Median log logit gain per rotary pair and denoising step over the 8 dev prompts (rows 0 to 7, seed 0),
on one shared diverging scale; static variants average each method's gains over prompts and steps.
(b) Dev prompt 0, seed 0: full frames and a native $384 \times 256$ crop. CLIP score, released 30.3; released
static 30.0 ($\Delta = -0.3$ [$-1.2$, 0.5]); paper equations 28.1 ($-2.2$ [$-3.6$, $-0.8$]); paper static
13.1 ($-17.3$ [$-21.0$, $-13.6$]). Differences are paired over the 8 dev prompts with 95\% bootstrap CIs
(2,000 resamples); ImageReward follows the same order.}

## F7 `f7_mechanism.pdf`
\caption{\textbf{The late-profile Constancy target stays off for the first 19 of 28 steps while the time-matched
target acts throughout, and attention-gain transfer is broad rather than band-selective.} (a) Mean
$|\log g - \log\tau|$ over spatial rotary pairs per step, median and interquartile band over the 8 dev prompts
(seed 0), both on the NTK base; below, per-pair correction maps. (b) Response of output-octave log power to a
1.5$\times$ logit gain on one octave (same-seed finite differences), mean of 3 prompts (seed 0), with power
measured along height and along width. Outlined cells are the diagonal (height 0.00 to 0.29, width $-0.22$ to
0.32); dots mark cells where all 3 prompts agree in sign.}

## F8 `f8_spectrum.pdf`
\caption{\textbf{Without SEGA's per-band allocation, luminance power at 8 to 32 pixel wavelengths sits 0.33 ln
units (about 40\%) above released SEGA's; Constancy-TM leaves a similar excess, and NTK falls 0.5 to 1.3 ln
units short from 8 to 1024 pixels.} (a) Signed log power error, $\ln P$ minus the mean $\ln P$ of 195
Aesthetic-4K references, per octave band of linear-sRGB luminance (radial power spectrum); (b) paired difference
from released SEGA. Held-out rows 8 to 194 ($n = 187$, seed 0); bands are 95\% bootstrap intervals over prompts
(2,000 resamples). The 8 to 32 pixel range is highlighted descriptively. All methods exceed the references at
2 to 8 pixels.}
