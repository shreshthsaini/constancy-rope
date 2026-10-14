# Sources and numerical conventions

Inspected on 2026-10-14. These are shallow source checkouts under
`/scratch/09032/saini_2/constancy_rope/third_party`; runtime code does not import
them. No model weights were downloaded for the port. Tests use the checkouts
only to execute isolated numerical definitions. If a checkout is absent, its
upstream parity tests are marked skipped.

| Source | Revision | Local directory | License found |
| --- | --- | --- | --- |
| [DyPE](https://github.com/guyyariv/DyPE) | `436fe612fb7695fa2d5dc32d63390c048ba6033c` | `DyPE` | MIT, copyright 2025 Guy Yariv |
| [Official SEGA](https://github.com/rajabi2001/sega), linked by its [project](https://rajabi2001.github.io/sega/) | `ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e` | `sega` | No LICENSE file in this revision |
| [UltraImage](https://github.com/thu-ml/DiT-Extrapolation/tree/ultra-flux), branch `ultra-flux` | `586ae882099ee768e9bbc6757118d572a1e71b25` | `UltraImage` | Apache-2.0 |

`diffusers==0.39.0` is the integration target. Its `FluxPosEmbed` provides the
base parity oracle and its native attention processor provides the identity
oracle. Diffusers is Apache-2.0 licensed.

## DyPE port

`crope/rope.py` adapts the frequency logic from
[`flux/transformer_flux.py`](https://github.com/guyyariv/DyPE/blob/436fe612fb7695fa2d5dc32d63390c048ba6033c/flux/transformer_flux.py).
The `dype` method corresponds to upstream `method="yarn", dype=True`.
Each spatial axis uses its own token count divided by 64. The first ramp blends
PI with NTK using rotation thresholds `(1.25, 0.75) * sigma**2`. The second
blends that result with base frequencies using `(16, 2) * sigma**2`. We preserve
the upstream integer correction bounds and float32 ramps. Axis 0 is unchanged.
The cosine/sine arrays retain the upstream spatial amplitude
`1 + 0.1 * ln(axis_scale)`, which scales q and k and therefore squares in the
attention logits. `embedding_gain()` exposes this amplitude for logging; do
not add another YaRN temperature controller to DyPE.

The only endpoint extension is `sigma=0`, where the upstream helper divides by
zero. We return base frequencies at that endpoint and retain its fixed spatial
amplitude. The default flow scheduler does not evaluate the model at its terminal
zero sigma. Positive-sigma behavior is tested directly against upstream.

The design's separate `yarn` method uses its stated wavelength ratio
`r = T_d / 64`, with a linear ramp from 1 to 32. Its global logit temperature
is applied by the gain controller. This is a literal reading of design.md,
which differs from the usual YaRN rotation-count convention `r = 64 / T_d`
and from DyPE's two-ramp YaRN implementation. For rectangular images the
global-temperature controller uses the geometric mean of per-side scales;
the per-axis frequency ramps remain independent. These choices are logged.

The `ntk` and `ntk2` methods use the explicit design bases
`theta * s**(D/(D-2))` and `theta * s**(2*D/(D-2))`. They do not inherit DyPE's
time schedule. `pi` divides each spatial frequency by its own side scale.

## SEGA variants and incompatibility with the supplied reference

The official implementation was found and inspected in
[`flux_sega/transformer_flux.py`](https://github.com/rajabi2001/sega/blob/ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e/flux_sega/transformer_flux.py).
It cannot match the supplied `sega_stats.modulator` on identical inputs because
the algorithms differ. The requested `sega` name retains design.md and the
explicit reference-parity contract. `sega_official` is an additional baseline
with independently implemented numerical formulas in `crope/sega_official.py`.
Its tests compare the per-axis profiles, radial profile, spread, allocation,
and complete scaled cosine/sine arrays against the released code.

| Choice | Design/reference `sega` | Released `sega_official` |
| --- | --- | --- |
| Spectrum input | Packed 64-channel latent `x_t` | `x_embedder(x_t)`, normally 3072 projected channels |
| Axis energy | Two-sided FFT2 marginal with Nyquist excluded | One-sided 1D FFT, average over other axis, rebinned with DC and Nyquist included |
| Radial energy | Integer-radius rings `1..L/2-1` | Radius normalized to the corner, rebinned with DC included |
| RoPE-to-spectrum map | Physical bin `L/T_d` with clipping | Normalized log-period rank over the full profile |
| Standardization | Population standard deviation plus `1e-8` | Sample standard deviation clamped to `1e-8` |
| Allocation | `m_ref * (1 - spread * centered_tanh(z))` | `m_ref * (1 - .15 * spread * centered_tanh(1.5*z))` |
| Minimum q/k amplitude | No extra minimum beyond the design | Clipped to 1 |
| NTK base multiplier | `s**(2*D/(D-2))` per spatial axis | `s**(2*D/(D-2))/(1+.1*ln(s))`, one longest-side scale |
| Reference amplitude | `s**.08` per design scale convention | `max(s,1)**.08`, longest-side scale |
| Axis 0 amplitude | `m_ref` | 1 |

Both use a flatness-derived spread with exponent 1.5, but their different
radial profiles change that spread. The official controller runs once on the
projected image features before transformer blocks and applies its full q/k
gain after RoPE to every token. Its companion `sega_official` RoPE variant
supplies the released NTK frequencies without duplicating the gain.

The inspected SEGA revision has no repository license file. We do not vendor
its source; the optional implementation expresses its numerical behavior and
records its provenance. The original source is needed only for parity tests.

Current upstream DyPE and SEGA launch scripts default to FLUX.1-Krea-dev.
This experiment intentionally uses the requested FLUX.1-dev checkpoint for
every method. Numerical parity of the embedding/controller does not establish
image-quality reproduction on a different checkpoint.

The upstream DyPE and SEGA pipelines also pass the target image-token count as
`max_seq_len` when calculating the scheduler shift. This holds `mu=1.15` at all
target sizes. The requested unchanged diffusers scheduler instead extrapolates
to `mu=11.55` at 4096x4096. Its final evaluated sigma is approximately 0.99974,
whereas the upstream setting reaches approximately 0.10472 with 28 steps.
`scripts/diagnose_schedule.py` exposes this difference without loading weights;
sampling retains the requested default. RoPE parity does not imply pipeline parity.

Reference SEGA's integer FFT-bin assignment uses float64 `fftfreq` coordinates
followed by truncation. We retain that convention even at non-power-of-two token
lengths such as 192, because rounding would change the supplied reference.

## UltraImage inspection

The official `ultra-flux` branch contains
[`flux/rope/rope.py`](https://github.com/thu-ml/DiT-Extrapolation/blob/586ae882099ee768e9bbc6757118d572a1e71b25/flux/rope/rope.py)
and Flux inference/attention code. It includes extrapolation, PI, NTK,
frequency selection, YaRN, and RIFLEx helpers. Its YaRN helper uses a fixed
1328-token rotation-count expression, unlike this experiment's stated 64-token
axis grid. No UltraImage source or attention kernels were copied. UltraImage
is not an additional campaign method in this implementation.

## Figure style

The plotting helper `_figure_style.py` is vendored from the installed
`orx-figures` skill, sourced from alphaXiv/openresearch-cli at commit `7c3ec36`.
Its MIT notice is in `FIGURE_STYLE_LICENSE.txt`. Prose punctuation was adjusted
to the account's writing rules; plotting behavior is unchanged.

## DyPE license notice

MIT License

Copyright (c) 2025 Guy Yariv

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
