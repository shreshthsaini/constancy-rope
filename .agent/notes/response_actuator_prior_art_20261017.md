# Targeted prior art: calibrated attention actuation

Checked October 17, 2026. This extends `response_prior_art_20261017.md`
with four material sources. It does not repeat that note's SEGA, SCG, JAPS,
denoiser-spectrum training, Spectral Guidance, or Fisher-Preserving comparisons.
No experiment was launched and no manuscript, implementation, or protocol was
edited. CR17 efficacy is not assumed in this assessment.

## Finding

AFM is the closest new attention/Fourier collision. It already establishes
attention-frequency intervention and explicitly discusses nonmonotonic response.
Fast constrained sampling establishes finite-difference diffusion correction
motivated by approximate response inversion. Neither inspected algorithm
identifies the particular low-dimensional attention-gain-to-clean-prediction
Fourier-power map proposed here. This is a narrow implementation distinction,
not evidence of novelty or efficacy. Generic least squares, trust regions,
signed derivatives, and local model reuse are not contributions by themselves.

## Verified primary sources

### 1. Attention Frequency Modulation: Training-Free Spectral Modulation of Diffusion Cross-Attention

Seunghun Oh and Unsang Park. **arXiv:2603.28114v1**, March 30, 2026;
v1 is the latest version listed when checked.
[Metadata](https://arxiv.org/abs/2603.28114),
[full method](https://arxiv.org/html/2603.28114v1).

Sections 3.2-3.3 and Algorithm 1 measure radial Fourier power of attention
concentration maps, then edit each token's spatial cross-attention logit map
before softmax. Low/high band gains follow denoising progress, optionally gated
by attention entropy. Equation 18 prescribes gains; it does not fit an
actuator-response matrix or solve a spectral-error inverse problem.
The paper explicitly states that logit-band scaling need not change the
post-softmax spectral statistic monotonically. Its diagnostic measures token
competition, not image Fourier content. The main intervention targets U-Net
encoder cross-attention, not rotary-pair multiplier allocation.

**Collision:** training-free frequency-selective attention edits, spectral
diagnostics, scheduled gains, adaptive entropy gating, and nonmonotonic
attention response. A broad claim that attention spectral control is new
would be untenable. **Distinction to test:** a measured signed response from
constrained rotary gains to clean-prediction power, used to select corrections.
AFM makes that measurement more necessary; it does not establish its usefulness.

### 2. Fast constrained sampling in pre-trained diffusion models

Alexandros Graikos, Nebojsa Jojic, and Dimitris Samaras.
**arXiv:2410.18804v3**, October 6, 2025; first version October 24, 2024.
v3 is the latest listed version.
[Metadata](https://arxiv.org/abs/2410.18804),
[full method](https://arxiv.org/html/2410.18804v3),
[author implementation](https://github.com/cvlab-stonybrook/fast-constrained-sampling).

Section 3 derives the desired Newton direction from `J e_t = e`, then uses
`J e` as a cheap approximate direction. A forward difference estimates this
denoiser-input Jacobian product. Algorithm 3 repeats constraint updates at
each denoising step. Appendix A.3 compares exact/pseudoinverse Newton,
approximate Newton, and gradient descent; Appendix A.5 tests difference sizes.
The full method inspected does not build a reusable attention-parameter
response matrix. Its noisy-latent actuator has the denoiser input dimension.

**Collision:** response-aware constraint correction, forward-pass numerical
derivatives, approximate inversion, and step-size validation already occur in
diffusion sampling. **Boundary:** our gain-to-statistic matrix is rectangular;
their argument for using a denoiser Jacobian as an approximate inverse does
not justify replacing its pseudoinverse by itself. Report fitted versus
realized residual reduction instead of importing their convergence argument.

### 3. Spectral Evolution Search: Efficient Inference-Time Scaling for Reward-Aligned Image Generation

Jinyan Ye, Zhongjie Duan, Zhiwen Li, Cen Chen, Daoyuan Chen, Yaliang Li,
and Yingda Chen. **arXiv:2602.03208v1**, February 3, 2026;
v1 is the latest listed version.
[Metadata](https://arxiv.org/abs/2602.03208),
[full method](https://arxiv.org/html/2602.03208v1).

Sections 4-5 optimize initial-noise low-frequency wavelet coefficients with
cross-entropy search while holding high-frequency coefficients fixed. Candidate
rewards use complete denoising trajectories. Frequency-banded initial-noise
perturbations measure final-output sensitivity. Their analytic propagation
model assumes approximately diagonal Fourier dynamics and an idealized
Wiener response; it is not a measured cross-band attention-control matrix.

**Collision:** identifying effective spectral control directions and allocating
inference effort to them. **Boundary:** initial-noise perturbations and
attention-gain perturbations are different actuators. Our off-diagonal or
negative measurements cannot refute their input-perturbation model. Conversely,
their initial-noise results do not establish that high-frequency rotary gains
have low control authority. Their use of final outcomes also emphasizes the
gap between our instantaneous prediction response and trajectory efficacy.

### 4. Spectral Conditioning of Attention Improves Transformer Performance

Hemanth Saratchandran and Simon Lucey. **arXiv:2603.07162v1**,
March 7, 2026; v1 is the latest listed version. The arXiv comments identify
NeurIPS 2025; this memo does not independently verify that venue label.
[Metadata](https://arxiv.org/abs/2603.07162),
[full method](https://arxiv.org/html/2603.07162v1).

Section 3 analyzes attention derivatives with respect to query/key/value
projection matrices and bounds Jacobian conditioning. Definition 3.6 and the
implementation add fixed correction matrices to those projections during
training. The weight matrices remain trainable; correction matrices stay fixed.
Here spectral conditioning concerns singular values, not spatial Fourier
power. No measured frozen-diffusion gain-to-power control loop appears in the
method inspected.

**Collision:** attention-parameter Jacobian analysis and modifying projection
spectra to improve conditioning. **Boundary:** this is a training intervention
with a different response object. Avoid calling our proposal simply
"spectral conditioning of attention"; that name obscures both distinctions.

## Actionable claim boundary

The exact candidate object remains
`R_bj = d S_b[x_t - sigma*v_theta(x_t,t,c;m(u))] / d u_j`, with state,
time, text, and the multiplier normalization rule fixed. Band statistics,
control directions, and the metric used for inversion must be declared.

1. **Predict before controlling.** A sign or off-axis response is not enough.
   Test an unfit perturbation magnitude and direction, including a combination
   of controls. Compare predictions against zero response and diagonal/scalar
   fits. Central differences identify a local derivative, not a global law.
2. **Separate signed response from extra freedom.** A full inverse has more
   degrees of freedom than scalar correction. Compare full, diagonal,
   sign-only, and shuffled maps under the same gain budget and forward-call
   accounting. Define cross-axis coupling using the actual output bins and
   normalized control coordinates; RMS redistribution already couples pairs
   within an axis by construction.
3. **Reuse is an empirical property.** Freeze a map from development states,
   then measure prediction error on other prompts and times before reusing it
   in generation. Compare reused and freshly identified maps where feasible.
   Amortization is not established by smooth plots, nor is caching a local
   response model a new algorithmic primitive. Count calibration cost in the
   break-even number of generations.
4. **Connect to resolution extrapolation.** The useful hypothesis is that a
   reproducible mismatch between a prescribed gain allocation and its actual
   spectral effect explains a specific extrapolation failure, and that cheap
   calibrated correction improves final generation. Neither a local residual
   reduction nor successful reference-spectrum matching establishes this.

If CR17 only shows that gains change power, the result adds little beyond
existing intervention diagnostics. If a predictable signed map improves
correction over equal-cost prescribed and scalar controls, it supports a
bounded mechanism study. Transfer, final-image quality, and competitive
resolution results remain separate requirements.

## Search limits

This was a bounded targeted search using combinations of attention/RoPE gain,
Fourier or spectral power, response/Jacobian calibration, finite differences,
system identification, feedback, diffusion resolution extrapolation, and
amortized control. Four primary methods were inspected beyond the previous
memo. No exact attention-gain-to-power identification-and-reuse algorithm was
located in those methods. This does not prove absence from the literature.
Unrelated physical-diffusion identification papers were excluded. A
calibration/caching search lead without inspectable full methods was not
promoted to a comparison. No broad novelty claim, first-use claim, or claim
that local-controller reuse is unoccupied follows from this search.
