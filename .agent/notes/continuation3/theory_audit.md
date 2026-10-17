# Theory revision memo

Audited 2026-10-17. Scope: retained `paper/sec/analysis_sega.tex`,
`background.tex`, and `method.tex`, checked against current `crope/gain.py`,
`spectral.py`, `sega_official.py`, and the pinned upstream SEGA source.
These sections remain excluded from the working report. No manuscript, code,
or experiment changes were made for this audit.

The useful theory is the algebra of gain application, an idealized invariance
result, and the forward-noising power identity. Claims about improved detail,
denoiser reliability, or closed-loop correction do not follow from that algebra.

## 1. SEGA slope invariance needs explicit conditions

**Exact idealized result.** For a nonconstant sampled log-profile `ell`, fixed
sampling locations, and a fixed standard-deviation convention, the ideal score
`z=(ell-mean(ell))/std(ell)` is unchanged by `ell -> a + beta*ell`, `beta>0`.
Linear interpolation of the log profile commutes with this affine map.
The centered tanh allocation is consequently unchanged. This is an elementary
standardization identity, not a new theorem about generative behavior.

**Implementation qualifications.** The reference `sega_modulator` uses
`log(E+1e-30)` and population standard deviation plus `1e-8`. If `s=std(ell)`,
its transformed score is `beta*(ell-mean)/(beta*s+epsilon)`, not the original
score. Approximate invariance requires both power values and standard
deviations to be well above their stabilizers. Released SEGA floors energy at
`1e-8` and divides by `max(sample_std,1e-8)`. It has the ideal invariance only
when the original and transformed profiles avoid both floors, up to floating
point error. Constant profiles are outside the ideal theorem's domain.
For 28 pairs, sample-standardized z equals population-standardized z times
`sqrt(27/28)` before stabilizers. [Pinned source](https://github.com/rajabi2001/sega/blob/ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e/flux_sega/transformer_flux.py),
[PyTorch std convention](https://docs.pytorch.org/docs/2.11/generated/torch.std.html).

The power-law corollary requires strictly positive frequency coordinates.
Released SEGA samples a DC-containing bin, where `k^-alpha` is undefined.
State the corollary on a positive-frequency profile or specify a positive
coordinate offset. State the transformation at the already formed profile:
raising a 2D spectrum to a power need not commute with marginalization/rebinning.
The result concerns allocation shape, not the full gain: the radial spread can
change. Replace "rank-like" with "standardized log-profile shape"; arbitrary
monotone transformations preserve ranks but do not preserve these z-scores.

## 2. Gain normalization is not logit-distribution normalization

**Exact.** A common multiplier on both coordinates of each rotary pair in q
and k scales that pair's dot-product contribution by `g_d=m_d^2`. For fixed
content, RoPE gives a finite trigonometric sum in relative position. A positive
scalar multiplying the entire logit row preserves ranking and decreases
softmax entropy as it increases. Per-pair gains have no corresponding monotonic
locality guarantee. The signs/phases of the content coefficients matter.
[RoFormer](https://arxiv.org/abs/2104.09864).

For an unclipped SEGA multiplier `m_d=m_ref*(1+a_d)` with `mean(a)=0`,
`mean(m_d)=m_ref`, but `mean(m_d^2)=m_ref^2*(1+mean(a_d^2))`.
Clipping at one changes both moments. This explains the CR14 RMS controls.
Per-axis RMS normalization preserves mean squared multiplier, not measured
logit variance, q/k norm, entropy, or a complete global temperature. Axis 0
remains unit-scaled and the two spatial axes need not have identical RMS.
Do not describe the experiment as exact entropy matching.

The log-token temperature is a prior heuristic/statistical approximation, not
a universal consequence of more keys. Different score distributions require
different scaling laws. Restore the Jin attribution with this qualification.
[Jin et al.](https://arxiv.org/abs/2306.08645),
[InfoScale](https://arxiv.org/abs/2501.08570),
[critical inverse-temperature scaling](https://arxiv.org/abs/2605.12697).

**Unsupported causal claim.** Increasing high-frequency rotary coefficients
does not prove a local-attention bias, increased image-band energy, improved
sharpness, or reduced tiling. Label those as hypotheses tested by the probes.
The recorded Krea transfer is anisotropic and the held-out interventions fail.

## 3. Forward-noise power is exact in expectation; reliability is heuristic

**Exact under stated assumptions.** For independent data `x0` and unit white
noise `epsilon`, `x_sigma=(1-sigma)*x0+sigma*epsilon` gives the manuscript's
expected raw-DFT power identity because the cross term averages to zero.
With the current orthonormal FFT and mean-per-band density, the white-noise
expectation is `C` on every retained band; hence the reference is `sigma^2*C`.
Spatial centering removes DC, which the retained axial bands exclude.

This is an ensemble expectation, not an exact decomposition of one realized
latent. A periodogram fluctuates even for pure noise; clipping the excess at
zero biases it upward. Thus "all gates near zero at sigma approximately one"
is not a theorem and requires measurement.

The ideal learned velocity is a conditional expectation of interpolation
velocities. Its ODE shares the interpolation's marginals under the theory's
assumptions, but an individual ODE trajectory need not be a straight line
between its own endpoints. At the ideal conditional predictor,
`x-sigma*v*(x,sigma)=E[x0 | x_sigma=x]`; it is not the true sample endpoint.
Extrapolated resolution, guidance, finite-step integration, and an approximate
model further limit that interpretation. [Rectified Flow, authors' exposition](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html),
[Flow Matching](https://arxiv.org/abs/2210.02747).

Therefore `rho=max(E_observed/(sigma^2*C)-1,0)` is a spectral excess-power
statistic inspired by forward SNR. It does not estimate posterior uncertainty
or validate the previous denoiser prediction's reliability. `rho/(1+rho)` has
the algebraic form of a Wiener factor, but its use as a controller gate is a
design choice. Delete the causal assertion that below-floor predicted power
must come from the prior rather than the sample. Human-vision analogies remain
motivations, not a derivation of the model's inverse response.

## 4. Other exact properties worth retaining, with smaller claims

- **Projection:** averaging an affine token embedding and removing its spatial
  mean is exactly one fixed linear projection of the latent channels. The
  quoted 1.7% power fraction needs its denominator and projection normalization
  defined. Low amplitude alone is not evidence of little information: rescaling
  a projection changes power and largely cancels in SEGA's statistics. Do not
  transfer a FLUX.2 channel-mean measurement to the learned Krea projection.
- **Rank map:** the stated index-linear mapping follows for geometric rotary
  frequencies with base greater than one, inactive frequency floor, and a
  nondegenerate log-period span. Include these conditions. Say it is not a
  physical wavelength-to-frequency match; "unrelated" overstates the result.
- **Released bounds:** `1 <= m_d <= 1.3*m_ref` remains a valid loose bound when
  `m_ref>=1`. At 4K the logit upper bound is about 2.11. The reference variant
  also has a finite upper bound; it lacks a positive lower bound and its signed
  multiplier can be negative. Its implementation uses the absolute value,
  which preserves `m_d^2` when the same sign acts on q and k. Replace "neither
  bound" with this precise distinction.
- **Constancy zero correction:** matching every positive band power to the
  anchored profile gives zero correction algebraically. Multiplicative scale
  invariance additionally requires no power floor to activate, fixed anchor
  membership, and unchanged controller inputs other than that prediction.
  This is a zero of the correction rule, not a stable fixed point of sampling.
  SEGA can also have zero tilt, for example with a flat profile. Delete the
  categorical claim that it has no such point.

## 5. Stale implementation claims must be rewritten before inclusion

| Retained text | Current implementation / required correction |
| --- | --- |
| Raw FFT band sums and grid-dependent noise counts | `axis_power` uses orthonormal FFT, averages the other axis and band frequencies, and sets the floor to `C`. |
| Piecewise-constant assignment; empty band leaves pair 7 inactive | Corrections interpolate in log frequency across retained bands; unresolved frequencies below `1/L` alone are masked. |
| Anchor always requires gate above 0.5 | Time-matched variants use the fixed anchor range without this visibility threshold; their empty-anchor fallback also differs. |
| Late native target for every experiment | Time-resolved profiles are used in `constancy_tm` and SEGA composites. |
| Current-time target is matched to a previous-step prediction | This is a timing mismatch in the existing B3 rule; CR14 `sega_ctm_lag` explicitly tests the previous prediction's sigma. No efficacy claim yet. |
| All variants retain log-N temperature and the same index-axis gain | SEGA composites retain released gains and axis 0 equals one. |
| Official static control is not implemented | `sega_official_static` exists and was tested at pilot scale. |
| Forward-noised FLUX.2 results characterize released SEGA | Those measurements concern the reference variant on a different latent space and forward process. Keep them as historical diagnostics only. |

In the comparison table, label the first column "our paper-inspired reference
implementation." Numerical FFT bins, epsilon, std convention, and index-axis
handling were not all explicit mathematical requirements in the SEGA paper.
Do not imply that every implementation choice is an author-specified equation.
[SEGA paper, Sections 3 and 4](https://arxiv.org/html/2605.22668v1).

Restore a short audited theory section first: pairwise logit scaling, the
qualified affine-invariance identity, and the normalized forward-noise
expectation. Keep physiological derivations and the old toy tables excluded
until their provenance and limited interpretation are stated beside them.
