# Rotary kernels and invariant spectral sensors: targeted prior art

Checked October 17, 2026. Read current handover, six named methods, selected
primary neighboring work, and relevant released code. Only this memo was
written. No jobs, model loads, implementation changes, or frozen-file edits.
The proposed FLUX residual-shift symmetry is under a separate architecture
audit; the conditional argument below does not declare that audit passed.

## Main finding and closest collision

The sharper hypothesis is that SEGA observes a feature component discarded by
the underlying prenorm network. This is more specific than arbitrary channel
basis dependence. However, **AReA already rejects channel averaging because of
opposite-sign cancellation and uses a shift-invariant channel-standard-deviation
sensor**. Its Appendix C.1, Equation 12 and released `detail_map` agree.
Thus channel cancellation, replacing the mean, and shift-invariant detail
sensing are already nearby prior art. A SEGA-specific causal failure may still
be useful, but its proof and repair must go beyond those claims.
[AReA paper](https://arxiv.org/html/2610.17227v1),
[pinned code](https://github.com/KooroshRH/AReA/blob/dcd4bcb114dbd9bac85e7178617c1c1f760ef760/area/tiling.py#L82).

## What the named methods actually change

| Method and verified version | Mechanism, measurement, and collision boundary |
| --- | --- |
| **AReA: Adaptive Receptive-Field Attention for Efficient High-Resolution Diffusion Transformers**, Javad Rajabi, Koorosh Roohi, David B. Lindell, Babak Taati; 2610.17227v1, October 12, 2026. [Full method](https://arxiv.org/html/2610.17227v1). | Early queries attend to representative pooled K/V tokens on a coarse shared coordinate grid. Later attention uses shrinking local windows, with content-dependent splitting from a Haar detail map. That map transforms per-token channel standard deviations, averaged over batch. It does not estimate rotary-pair Q/K covariance or fit a displacement-response kernel. Generic receptive-field schedules, coarse coordinate remapping, and an invariant scalar feature sensor collide directly. |
| **SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers**, Javad Rajabi, Kimia Shaban, Koorosh Roohi, David B. Lindell, Babak Taati; 2605.22668v1, May 21, 2026. [Full method](https://arxiv.org/html/2605.22668v1). | Section 4.1 explicitly takes channel mean before Fourier power. Section 4.2 standardizes log spectral samples and sets rotary-pair amplitudes. The paper's radial spatial-rotation invariance is unrelated to channel-basis invariance. Neither per-pair activation energy nor actual logit covariance is normalized. Released behavior also averages across batch, whereas the paper says samples are handled independently. With batch one, this extra discrepancy is inactive. |
| **DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion**, Noam Issachar, Guy Yariv, Sagie Benaim, Yossi Adi, Dani Lischinski, Raanan Fattal; latest 2510.20766v2, January 29, 2026. [Full method](https://arxiv.org/html/2510.20766v2). | Time-dependent PI/NTK/YaRN frequency or threshold schedules restore higher rotary frequencies later in denoising. Dy-YaRN also includes uniform attention sharpening. The schedule is motivated by spectral progression, not fitted from current pair covariance. Changing spatial phase frequencies or scheduling them alone is occupied. |
| **HRDiT: Training-Free High-Resolution Image Generation with Off-the-Shelf Diffusion Transformer Models**, Yu Xue, Haoxuan Qu, Zhuoling Li, Hongbin Xu, Jianxiong Yin, Simon See, Hossein Rahmani, Jun Liu; 2608.07003v1, August 7, 2026. [Full method](https://arxiv.org/html/2608.07003v1). | Spatial Position Alignment replaces token coordinates by bundle indices and averages attention contributions across sliding bundle boundaries. Head-adaptive pruning uses a preparatory scope assignment. This is an explicit displacement/position correction plus calibrated head scope. It does not normalize rotary-pair contribution covariance. A proposed phase remapping or reusable head-locality calibration must distinguish these mechanisms. |
| **TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers**, Yihua Liu, Fanjiang Ye, Bowen Lin, Rongyu Fang, Chengming Zhang; 2603.08928v1, March 9, 2026. [Full method](https://arxiv.org/html/2603.08928v1). | Adds a scale-dependent text-logit bias and raises temperature toward one over denoising. The text-mass argument assumes image-token exponential mass scales with token count; it is not exact content-dependent partition matching. This covers token-balance correction and scheduled global sharpening, without rotary-pair energy or covariance normalization. |
| **Attention Frequency Modulation: Training-Free Spectral Modulation of Diffusion Cross-Attention**, Seunghun Oh and Unsang Park; 2603.28114v1, March 30, 2026. [Full method](https://arxiv.org/html/2603.28114v1). | Fourier-filters token-specific cross-attention logits before softmax, with time and optional entropy gates. Its post-softmax attention-spectrum statistic need not change monotonically with logit-band gain. It already covers adaptive spectral attention editing, but neither projected-feature channel statistics nor rotary-pair covariance calibration. |

The latest versions above were checked against arXiv submission histories.
These summaries describe methods, not reproduced rankings or SOTA claims.

## Released-code checks

- SEGA current commit `ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e`:
  [`transformer_flux.py` lines 263-267 and 290-296](https://github.com/rajabi2001/sega/blob/ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e/flux_sega/transformer_flux.py#L263)
  average batch and channels before FFT. Line 691 applies `x_embedder` before
  the controller sees features. Local `crope/sega_official.py` matches this.
- AReA current commit `dcd4bcb114dbd9bac85e7178617c1c1f760ef760`:
  [`tiling.py` lines 82-90](https://github.com/KooroshRH/AReA/blob/dcd4bcb114dbd9bac85e7178617c1c1f760ef760/area/tiling.py#L82)
  uses channel std, then batch mean and spatial centering.
  [`backbones/flux.py`](https://github.com/KooroshRH/AReA/blob/dcd4bcb114dbd9bac85e7178617c1c1f760ef760/area/backbones/flux.py#L23)
  preserves backbone Q/K normalization and applies positional transforms after
  pooling. No pair-covariance equalizer was found in the inspected path.
- HRDiT current commit `f3be935da797c7dc98ecff07cf4a5f01e1ddb51d`:
  [`spa.py`](https://github.com/zylwithxy/HRDiT/blob/f3be935da797c7dc98ecff07cf4a5f01e1ddb51d/hrdit/spa.py#L12)
  implements bundle-coordinate variants;
  [`attention.py`](https://github.com/zylwithxy/HRDiT/blob/f3be935da797c7dc98ecff07cf4a5f01e1ddb51d/hrdit/attention.py#L52)
  retains backbone Q/K normalization.
- DyPE project-pinned commit `436fe612fb7695fa2d5dc32d63390c048ba6033c`:
  [`transformer_flux.py`](https://github.com/guyyariv/DyPE/blob/436fe612fb7695fa2d5dc32d63390c048ba6033c/flux/transformer_flux.py#L640)
  multiplies both sine and cosine by the YaRN multiplier. This scales Q/K
  amplitudes and therefore squares the multiplier in each logit contribution.
  This is a checked project revision, not a claim about current upstream HEAD.

No author implementation was verified for TIDE or AFM in this bounded pass.

## Exact algebra: three different normalization questions

These are elementary identities, not proposed new theorems.

**Feature spectrum.** For channel-vector Fourier coefficient `z_hat(f)`, SEGA's
scalar-map power is `|1^T z_hat / C|^2`. It is one rank-one quadratic form,
including cross-channel cancellation. Channel-averaged power instead equals
`||z_hat||^2/C`, the trace of its cross-spectral matrix. The latter is invariant
to orthogonal channel re-expression but not to arbitrary channel rescaling.
The fixed frozen network does not generally admit arbitrary orthogonal
residual-stream rotations without additional parameter changes.

Let `P = I - 11^T/C`. Under a token-dependent common-channel shift
`z_i -> z_i + a_i*1`, the channel mean changes by `a_i`. Spatial centering only
removes the spatially constant part of `a`. Ordinary channel-summed power is
also affected: `||z_hat||^2 = ||P z_hat||^2 + C*|mean(z_hat)|^2`.
In contrast, `||P z_hat||^2` is invariant to this shift. Per-token channel
centering commutes with spatial FFT, so it can be performed before the FFT.
Our existing `crope/spectral.py` already computes channel-summed power but does
not thereby remove this common-channel component. A total-power replacement
alone is not the proposed shift-invariant repair.

**Q/K energy.** With rotary-pair amplitudes `m_d`, preserving `sum_d m_d^2`
preserves the gain-vector RMS. Query energy instead uses
`sum_d m_d^2 ||q_d||^2`, and key energy uses its own activation weights.
Neither implies fixed attention-logit variance or entropy.

**Content-dependent phase kernel.** For a pair and displacement angle `phi`,
`q^T R(phi) k = A*cos(phi) + B*sin(phi)`, where
`A=q1*k1+q2*k2` and `B=q2*k1-q1*k2`. Actual pair contributions depend on both
these content terms and displacement. Their aggregate variance is a quadratic
form in `g_d=m_d^2` with the covariance of pair contributions, including
off-diagonal terms. Pair norm products bound their amplitude but do not
determine the phase or signed contribution. Equalizing those norms can erase
learned anisotropy; it is not justified by a gain-RMS argument.

## Established symmetry and nearby kernel precedents

**LayerNorm shift symmetry is established.** Ba, Kiros, and Hinton,
*Layer Normalization*, [1607.06450, Section 5.1](https://arxiv.org/pdf/1607.06450),
analyze recentering invariance. TransformerLens implements residual-writing
weight centering for compatible prenorm architectures:
[official documentation](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.weight_processing.html),
[pinned source](https://github.com/TransformerLensOrg/TransformerLens/blob/bd55eab6737584685b921fdfcd15f191579e64b1/transformer_lens/weight_processing.py#L1176).
Its source explicitly treats post-norm embeddings differently. An appropriate
claim must be architecture-specific; the symmetry itself is not new.

Conditional on every non-residual read using LayerNorm and the terminal
readout removing the common shift, induction through prenorm residual blocks
gives `F(h+a*1,m)=F(h,m)` for fixed controller gains `m`. LayerNorm's additive
shift invariance holds in real arithmetic even with epsilon and fixed learned
affine parameters. Conditioning modulations preserve it when independent of
the shifted features. This does not establish BF16 numerical parity, and
clipping or an unnormalized branch can invalidate the induction.

The composite controller can break the symmetry if `m(h+a*1) != m(h)` and
the model responds to that gain difference. Noninvariant gains alone do not
prove an output failure. Calling the channel mean causally irrelevant inside
the unmodified model does not prove it is statistically uninformative about
content in the original checkpoint's chosen parameterization.

Other relevant primary boundaries:

- **Expected Attention: KV Cache Compression by Estimating Attention from
  Future Queries Distribution**, Alessio Devoto, Maximilian Jeblick, Simon
  Jegou; [2510.00636v1](https://arxiv.org/html/2510.00636v1), October 1, 2025.
  Models query mean/covariance and positional averaging to estimate key
  importance. Covariance-aware rotary attention statistics are established;
  its Gaussian and positional approximations do not supply a DiT correction.
- **WorldRoPE: Probabilistic World-Centric Rotary Embedding for Multi-View
  Attention**, Mijeong Kim, Kyujin Lim, Bohyung Han;
  [2610.16337v1](https://arxiv.org/html/2610.16337v1), October 12, 2026.
  Integrates rotary blocks over positional Gaussian mixtures, yielding
  covariance-dependent attenuation and effective phase. This directly occupies
  uncertainty-based per-pair phase/amplitude correction, though it predicts
  multi-view geometry rather than adapting a frozen high-resolution DiT.
- **When Local Variance Optimality Is Not Enough: RoPE-Aligned Q/K Rotations
  for Dynamic 4-Bit Quantisation**, Shuhan Wang, Yilin Luo, Nan Xu, Chi Wang
  Cheung; [2608.13365v1](https://arxiv.org/html/2608.13365v1), August 13, 2026.
  Calibrates pair covariance with deployed rotary phases for a quantization
  surrogate, with negative end-to-end results for the restricted pairwise
  method. Shared within-pair Q/K rotations commute with RoPE and preserve
  logits; applying such a rotation alone cannot repair a spatial kernel.

## One bounded causal question worth retaining

**Does SEGA's feedback path turn a verified null perturbation of the frozen
backbone into a measurable denoising change?** This question is narrower than
"channel means cancel" and has a decisive intervention. It was not answered
in the inspected methods. It remains a hypothesis pending the FLUX audit.

First compare the unmodified/fixed-gain model under a controlled spatial
common-channel shift, then compare adaptive SEGA on exactly the same pair of
states. Record feature-statistic change, gain change, and output change
separately, with a numerical control appropriate to the model dtype. Use
fixed-gain replay to distinguish the feedback path from rounding. Epsilon
does not mathematically break additive shift invariance, while floating-point
rounding can.

A shift-invariant alternative could observe centered multichannel power or the
actual normalized representation. These are standard statistics, not new
mechanisms by themselves. They differ from a scalar norm-map spectrum:
spatial features `h_i = sign_i*v` with `1^T v=0` can change orientation while
their token norms stay constant. A scalar norm map misses that variation;
the multichannel spectrum detects it. This elementary distinction is a useful
sensor diagnostic, not evidence that preserving it improves generation.

Only if the null-perturbation effect exists, the replacement removes it, and
the replacement also improves natural generation against strong controls is
there a possible method contribution. An artificial perturbation failure
alone establishes sensitivity to representation choice, not a natural-data
quality defect. Do not claim all hidden information is recoverable from
centered power, or that invariance guarantees semantic fidelity.

## Search limits and decision

The search targeted rotary-pair normalization, phase/displacement calibration,
SEGA channel aggregation, LayerNorm shift symmetries, and invariant inference
controllers. No exact SEGA/prenorm-null-sensor critique was located. The
specific conditional mechanism above is not claimed absent from all prior art.
The closest practical collision is AReA's sensor; the exact architectural
symmetry has older normalization and interpretability precedents. Generic
normalization, covariance fitting, or phase remapping should not be pursued
as novel on the strength of this search. The null-direction feedback audit is
the most explicit remaining causal question, with efficacy still untested.
