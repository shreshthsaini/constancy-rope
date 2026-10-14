# Design spec (source of truth for the Codex brief)

## Setting
FLUX.1-dev (diffusers FluxPipeline, diffusers 0.39), bf16, 28 steps, guidance 3.5, default flow-match scheduler (with its resolution-dependent shift). Train grid: 1024^2 px -> 64x64 tokens (VAE /8, 2x2 packing). Targets: 4096^2 (256x256 tokens, s=4 per side), 3072^2, 2048x4096.
Flux RoPE: head_dim 128, axes_dims (16, 56, 56), theta 10000, interleaved pairs (x[...,0::2], x[...,1::2]). Dims 0-15: axis 0 (text/time index, all zeros for image tokens), 16-71: height, 72-127: width. 28 frequencies per spatial axis.

## Position embedding variants (spatial axes only, per-axis s_a = target_tokens_a / 64)
- base: unchanged. pi: theta_d / s. ntk: b' = b * s^(D/(D-2)), D=56. ntk2 (SEGA's): b' = b * s^(2D/(D-2)).
- yarn: NTK-by-parts ramp on r_d = T_d / 64 (alpha=1, beta=32) + global logit temperature 0.1 ln(s) + 1, s = area ratio? -> use per-side s as in YaRN 1D form, log the choice.
- dype: port official DyPE (github guyyariv/DyPE) Flux implementation if available, else paper Eq.13 Dy-YaRN.

## q/k gain hook
After RoPE, multiply q and k (all tokens) by a per-dim vector m (len 128; equal values on each rotary pair). logit gain of band = m^2.

## SEGA (paper Eqs 3-7, arXiv 2605.22668)
Z = x_t token grid (64 ch), mean over channels, subtract spatial mean, P=|FFT2|^2. E_a: marginal over the other axis, bins 0..L_a/2-1. E_iso: ring average, r=1..L/2-1. For dim d on axis a: wavelength T_d = 2pi/theta'_d (after ntk2), bin k = L_a/T_d interpolated in log E, clipped to [1, L_a/2-1]. z-score over the 28 dims, s_d = tanh(z) - mean(tanh z). SF = geomean/mean of E_iso; sigma = 1 - SF^1.5. M_d = 1 - sigma s_d; m_d = m_ref M_d, m_ref = s^0.08 (s = per-side ratio, also log area variant). Axis-0 dims get m_ref.
If official SEGA code exists, use it instead and note it.

## SEGA-static (ablation of SEGA's own claim)
m_d = m_ref * mean over (calibration prompts, steps) of SEGA's M_d at the same resolution. No FFT at run time.

## Constancy RoPE (ours)
Calibration (once per model): native 1024^2 runs, M prompts, record per-axis half-octave power profile of x0hat = x_t - t*v (channel-SUMMED power) at each step; native profile P_nat(f) = mean log power over late steps (t<0.3), f in cycles/token.
Per step t at target res, per spatial axis a, half-octave bins b of |k|/L (cycles/token), only bins with >=1 FFT bin:
- E_t(b) = channel-summed axis-marginal power of x_t; noise floor n_b = (#k bins in b) * L_other * H * W * C_noise (unit-variance noise per channel; unnormalized FFT gives H*W per bin per channel; marginal sums L_other bins). rho_b = max(E_t/(t^2 n_b) - 1, 0); w_b = rho/(1+rho).
- E0(b) from x0hat of the previous step (first step: no correction).
- Target: log S*(b) = P_nat(f_b) + c, c = mean over anchor bins (wavelength 16..64 tokens and w_b>0.5) of log E0 - P_nat. For f below native range (wavelength > 64 tokens) extrapolate P_nat linearly in log-log from its two lowest bins.
- e_b = 0.5 * (log S* - log E0); log g_b = log tau_N + eta * w_b * clip(e_b, -1, 1); tau_N = ln(N_tgt)/ln(N_train) (image tokens).
- RoPE dim d gets the bin of its wavelength; wavelengths longer than the grid get log tau_N only. q,k scale = exp(log g / 2). Axis-0 dims: sqrt(tau_N).
Options: eta (default 1), gate on/off, target profile|slope, temperature logN|sref|none.
Ablations: temp_only (tau_N only), constancy w/o gate, constancy with SEGA's m_ref instead of tau_N.

## Metrics
Prompts: Aesthetic-4K eval captions (+ its 4K reference images). Main: 4096^2, 1 seed, as many prompts as budget allows (target 100+).
CLIP score (ViT-L/14), ImageReward, PickScore, HPS (v3 if easy), MUSIQ and CLIP-IQA (pyiqa) on full res and on 1024 crops, FID and KID (resized 299) and patch-FID/KID on random 1024 crops (fixed crop seed) vs Aesthetic-4K refs.
HVS metrics: per-octave log luminance power error vs reference set (slope error), tiling index (normalized autocorrelation of luminance at lags = 1024 px and 2048 px vs 1.5x those lags).
Bootstrap 95% CIs over prompts, paired differences vs SEGA.

## Kill tests
T1 SEGA-static ~= SEGA. T2 MTF probe: multiply gain of one octave by 1.5 throughout sampling at 4096^2, same seed; d log power_b / d log g_b must be >0 on the diagonal. T3 failure spectra of ntk2 at 4K.
