# Prior art for response-calibrated spectral attention

Checked October 17, 2026 from compute host `i618-051`. Read the current novelty
note and frozen response-probe protocol. This is a full-method comparison of
the named papers and selected close neighbors, not an exhaustive novelty search.
Only this memo was written. No jobs, manuscript edits, or frozen-file edits.

## Named papers: what they already do

**Graikos, Jojic, Samaras, arXiv:2609.36210v1, September 28, 2026.**
Sections 3.2 and 4, Algorithm 1, and Appendix D analyze the denoiser-input
Jacobian `J_x = d f_theta(x,t)/dx`. They approximate products by forward
differences and use block power iteration with QR to estimate leading
directions. Training perturbs noisy inputs: random directions suppress response;
the negative denoising residual encourages response along a data-related
direction. The learned model weights change. Their "spectrum" concerns
Jacobian eigenstructure, not Fourier power bins. Neither finite-difference
denoiser probes nor selective response amplification is new to our project.
Our local object instead differentiates a Fourier-power statistic with respect
to attention multiplier controls at fixed latent, time, and text. Their
reported association between larger eigenvalues and better generation gives
no reason to maximize this different actuator's response.
[Full paper](https://arxiv.org/html/2609.36210v1).

Code cross-check: current source revision
`399a622070bb7324bb55bdf2b8c3f931dd1c3052`,
[`imagenet/optimize.py`, lines 175-217](https://github.com/AlexGraikos/denoiser_jacobian_spectrum/blob/399a622070bb7324bb55bdf2b8c3f931dd1c3052/imagenet/optimize.py#L175),
uses a detached residual, a top-5-percent spatial mask, a perturbed second
forward pass, and moving response statistics. This confirms training-time
regularization. No execution or reproduction was performed.

**Hen et al., JAPS, arXiv:2511.18471v3, June 21, 2026.**
This is the latest version shown by arXiv. Sections 3.1-3.3 derive paired
proximal and direct likelihood surrogates, with the direct direction containing
the denoiser-input Jacobian transpose. Their correction is
`g = u * <v,u>/<u,u>`: project the proximal direction `v` onto the
Jacobian-preconditioned direction `u`. They explicitly investigate harmful
departures of learned Jacobians from ideal covariance properties. Appendix
B.3 uses autodiff JVPs/VJPs for response diagnostics. Thus using Jacobian
information while rejecting harmful correction components is already an
established diffusion mechanism. JAPS acts on noisy sample updates for inverse
problems with observations; it does not identify a map from attention controls
to Fourier-power changes. Our rectangular response matrix has no reason to be
symmetric, positive semidefinite, or a posterior covariance.
[Latest abstract/version history](https://arxiv.org/abs/2511.18471),
[full v3](https://arxiv.org/html/2511.18471v3).

Code cross-check: revision `3d6e3fe00669f83f55fb8dfbb699a74acc7f8507`,
[`guided_diffusion/condition_methods.py`, lines 560-573](https://github.com/liavhen/JAPS/blob/3d6e3fe00669f83f55fb8dfbb699a74acc7f8507/guided_diffusion/condition_methods.py#L560),
implements the stated projection, including the signed scalar. Its extra-cost
claim is relative to computing the direct surrogate, not to ordinary
unconditional sampling. No execution or reproduction was performed.

## Nearby mechanisms that constrain the claim

| Source checked | Existing mechanism and boundary |
| --- | --- |
| [Spectral Guidance, Moreira et al., ICML 2026](https://proceedings.mlr.press/v306/moreira26a.html); [Sections 4-5, Algorithms 1-2](https://arxiv.org/html/2605.28900v1) | Learns singular functions of a conditional-expectation operator with an auxiliary self-supervised network, estimates guidance coefficients from reference samples, and differentiates through that small network during sampling. It projects objectives into diffusion-dependent spectral coordinates. It requires offline training despite avoiding denoiser retraining. Our Fourier statistics and local attention actuator are different objects. Generic spectral projection, informative subspaces, or time-dependent controllability are insufficient novelty claims. |
| [Fisher-Preserving Guidance, Sections 3.2-3.3 and Appendix G](https://arxiv.org/html/2605.29937v1) | Computes condition-to-denoised-action sensitivity and projects an action-space task gradient to preserve a sensitivity statistic to first order, with an approximate factorization. This is a direct conceptual collision with constrained Jacobian-aware guidance, although the task is diffusion-policy navigation. Its claims about Fisher information/manifold preservation should not be imported as guarantees for attention-RMS preservation. |
| [Spectral Correction Guidance, Section 4 and Algorithm 1](https://arxiv.org/html/2610.02753v1) | Corrects intermediate latent Fourier magnitudes toward a reference spectrum while retaining phase. It already occupies feedback from spectral error to sampling correction. Our question is whether an indirect attention actuator can correct errors selectively and preserve quality better at acceptable cost. Merely replacing its actuator does not establish a substantive advance. |
| [SEGA, Sections 4.1-4.2](https://arxiv.org/html/2605.22668v1) | Maps observed spectral profiles to rotary-component gains with a prescribed allocation. The proposed experiment measures the model's response to those gains. A response-based inverse rule would differ from this allocation, but its benefit remains unmeasured. |

## What the current experiment identifies

Write the measured object explicitly:

`R_bj = d S_b[x_t - sigma*v_theta(x_t,t,c; m(u))] / d u_j at u=0`,

where `S_b` is log Fourier-power density and `m(u)` is the specified gain
perturbation. It is an instantaneous clean-prediction response. It is neither
the denoiser-input Jacobian nor a derivative of the final generated image.
The first named paper studies input sensitivity; the second transforms an
inverse-problem direction. Neither validates our local-to-final transfer.

For an RMS-preserving axis intervention with mask `h`, the actual tangent in
logit-gain space is `h - sum(m_d^2*h_d)/sum(m_d^2)` over that axis.
Consequently each nominal octave intervention also changes the other pairs
on its axis. This is a constrained redistribution derivative, not an isolated
band derivative. The code correctly describes preservation of multiplier
RMS; the protocol was corrected before GPU submission to describe gain-vector RMS
preservation. It does not preserve q/k activation norms.

Three controls reveal only three directions of response. They cannot establish
the full reachable spectral subspace. Negative entries are valid signed
sensitivities, not violations of a covariance theorem. Singular values of a
future rectangular response matrix quantify local conditioning; eigenvalue
and positive-semidefinite terminology is generally inappropriate.

## Novelty boundary and actionable next steps

**Standard components:** central finite differences, same-state interventions,
linear response fitting, ridge least squares, singular-value truncation,
projection, trust regions, and coefficient-moment constraints. Their assembly
is a plausible engineering approach. It is not itself an established new
mechanism, theorem, or competitive generation method.

**Potential research contribution, conditional on evidence:** demonstrate that
resolution extrapolation produces reproducible sign/cross-band mismatches
between spectral errors and the available attention actuator, then show that
a cheap transferable response model corrects those errors while preserving
semantics and image quality. The specific causal failure and the practical
benefit would need to exceed what prescribed gains and direct spectral
correction already achieve. This narrow review neither establishes originality
nor rules out a closer existing method.

1. Keep the frozen probe as a development measurement. Verify replay identity,
   half-step prediction error against zero change, and perturbation size after
   BF16 casting. A fitted derivative must predict an unfit intervention before
   it is used to justify inversion. Do not maximize response magnitude merely
   because denoiser-input eigenvalues correlate with quality in another model.
2. If predictive, separate control identification from target validity. A
   well-measured response can still steer toward the wrong scene spectrum.
   Test a known small, reachable spectral intervention first; measure whether
   the realized change agrees with the predicted change. This establishes
   controllability in the sampled directions, not perceptual usefulness.
3. Compare a future controller with sign-only, diagonal, full-response, and
   scalar-only controls at matched cost. Include a random/sign-shuffled
   response control to test whether the measured map contributes information.
   For low-rank claims, report sensitivity to rank and conditioning.
4. Test local-to-final transfer on development data. A one-call improvement in
   predicted clean power may disappear or reverse under subsequent sampling.
   If calibration is reused, test new prompts/times before calling it portable.
5. Count probe/calibration forward calls, model loads, and end-to-end latency.
   Keep SEGA and direct Fourier correction as mechanism comparators; retain
   current resolution-extrapolation baselines for eventual efficacy evaluation.
   Preserve fresh confirmation prompts until the method and claims are frozen.

The next defensible claim is about measured local response and prediction
accuracy. A successful probe alone is not a method-paper result.
