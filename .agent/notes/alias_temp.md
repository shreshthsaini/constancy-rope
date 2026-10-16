# Alias temperature and SEGA refinement

## Frozen specification for the second campaign

Both candidates use `CROPE_ROPE=sega_official`. Native image tokens form a
64 by 64 grid; a 4096 by 4096 image has 256 by 256 packed image tokens.
The released Krea checkpoint, guidance 4.5, 28 steps, seed 0, native scheduler
shift, prompts, and scoring procedure match the earlier campaign.

For spatial pair d on axis a, let theta_d be its angular frequency after the
SEGA RoPE base change, and lambda_d = 2 pi / theta_d in tokens. Use that same
lambda_d for both the native and target effective-pool calculations. Define

    n(lambda, L) = min(L, max(1, L / lambda))
    K_native,d = n(lambda_d, 64) * 64
    K_target,d = n(lambda_d, L_target,a) * L_target,other
    g_d = ln(K_target,d) / ln(K_native,d).

This applies the requested Jin-style log-pool rule separately to each pair.
The other-axis multiplier retains all keys on the axis that this pair does
not encode. It is one axial count times the other-axis length, not two
aliasing counts. Axis 0 uses ln(H_target W_target) / ln(64 * 64).

The derivation substitutes these effective counts into the global
log-pool scaling approximation. The proposed count is a grid-cell heuristic,
not an exact count of distinct rotary phases. Arbitrary softmax logits need
not preserve entropy under this gain. We test the proposed mechanism without
asserting an entropy-invariance theorem or a novelty result.

Limits for extrapolation:

- As lambda tends to zero, both counts saturate at their axis lengths.
  Thus g_d = ln(H_target W_target) / ln(64 * 64), the global ratio.
- If lambda >= L_target,a >= 64, both axial counts equal one. Thus
  g_d = ln(L_target,other) / ln(64), the other-axis-only ratio.
- At the native grid every gain equals one.
- At 4K on a square grid the two extreme limits both equal 4/3.
  Intermediate wavelengths can have larger gains. The allocation is
  therefore not a monotonic interpolation between different endpoints.

`alias_temp_sref` subtracts the mean log gain across all rotary pairs,
including axis 0, and adds 0.16 ln(s), where s is the longest-side scale
used by released SEGA (s=4 here). This makes the geometric mean logit gain
exactly s^0.16. Repeating each pair across its two coordinates leaves that
mean unchanged. This fixes the reference gain, not the actual geometric
mean of SEGA's clipped, input-dependent allocation.

Each q and k coordinate is multiplied by sqrt(g_d), so its dot-product
contribution receives g_d. The alias modes have no adaptive correction or
additional global temperature.

## SEGA plus time-matched constancy

`sega_ctm` first computes the released SEGA q/k multiplier m_d from the
current `x_embedder` output. The existing time-matched controller supplies

    delta_b = eta * w_b * clip(e_b, -1, 1)
    e_b = 0.5 * (native_log_power_b(sigma) + anchor_offset - log_power_b)
    g_total,d = m_d^2 * exp(interpolate(delta_b)_d).

Interpolation, visibility gate w_b, anchor normalization, and unresolved-band
mask are exactly those of `constancy_tm`. The target is evaluated at the
current scheduler sigma, while the clean estimate is from the preceding
transformer call, as in the existing controller. The first step has no
correction. Axis 0 retains released SEGA's unit gain. No extra temperature
is applied. The default eta is 0.5; eta 1.0 is the other registered variant.
The attention processor uses m_d * exp(delta_d / 2) on each of q and k.

## Selection and reporting

Dev prompts are rows 0 through 7, used only for selection. Compare the same
eight prompt IDs against pilot `sega_official`. Eligibility requires higher
mean ImageReward AND higher mean full-resolution MUSIQ AND CLIP delta >= -0.3.
At most two variants advance. If more than two pass, select the two largest
ImageReward deltas, breaking a tie with MUSIQ then the variant name.
Held-out generation starts at row 7 to reuse the established chunk structure;
analysis excludes row 7 and reports only rows 8 through 194, n=187.

No eligible dev variant means no held-out generation. Touch `taskq/STOP`
only after pending is empty and scoring is complete. Never submit or cancel
allocations from this session. Dev scores are not held-out results.
