"""Step-level spectral gains and the diffusers 0.39 FLUX attention processor."""

from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F

from .spectral import NativeProfile, axis_power, log_interpolate


def sega_modulator(mean_latent, theta, theta_height=None):
    """Return (width, height) M and sigma, matching sega_stats.modulator.

    Rectangular grids use each axis length and circular rings up to min(H,W)/2.
    This is the design/reference algorithm. See sega_official for released code.
    """
    x = mean_latent
    if x.ndim != 2 or min(x.shape) < 4:
        raise ValueError("SEGA expects a two-dimensional map with sides >=4")
    x = x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
    h, w = x.shape
    p = torch.fft.fft2(x - x.mean()).abs().square()
    # NumPy's reference constructs fftfreq in float64, then truncates to int.
    # Float32 coordinates misassign bins at non-power-of-two sizes such as 192.
    ky = (torch.fft.fftfreq(h, device=x.device, dtype=torch.float64).abs() * h).to(torch.int64)
    kx = (torch.fft.fftfreq(w, device=x.device, dtype=torch.float64).abs() * w).to(torch.int64)
    radius = torch.sqrt(ky[:, None].to(x.dtype).square() + kx[None].to(x.dtype).square()).round().long()
    size = int(math.ceil(math.hypot(h // 2, w // 2))) + 1
    sums = torch.zeros(size, device=x.device, dtype=x.dtype).scatter_add_(0, radius.flatten(), p.flatten())
    counts = torch.bincount(radius.flatten(), minlength=size).to(x.dtype)
    iso = (sums / counts.clamp_min(1))[1:min(h, w) // 2]
    sf = (iso.add(1e-30).log().mean().exp() / iso.mean().clamp_min(1e-30)).clamp(0, 1)
    sigma = 1 - sf.pow(1.5)
    out = []
    theta_height = theta if theta_height is None else theta_height
    for length, k, marginal, frequencies in ((w, kx, p.sum(0), theta),
                                            (h, ky, p.sum(1), theta_height)):
        e = torch.zeros(length // 2 + 1, device=x.device, dtype=x.dtype).scatter_add_(0, k, marginal)
        e = e[:length // 2].add(1e-30).log()
        kd = (length * frequencies.to(x) / (2 * math.pi)).clamp(1, length // 2 - 1)
        lower = kd.floor().long()
        upper = (lower + 1).clamp_max(len(e) - 1)
        loge = e[lower] + (kd - lower) * (e[upper] - e[lower])
        z = (loge - loge.mean()) / (loge.std(unbiased=False) + 1e-8)
        tilt = z.tanh() - z.tanh().mean()
        out.append(1 - sigma * tilt)
    return torch.stack(out), sigma


def alias_logit_gain(wavelength, length, other_length, native_length=64, native_other=64):
    """Specified effective-pool heuristic, with the same post-base wavelength in both pools."""
    # n(lam,L) = min(L, max(1, L/lam)); axial pairs leave the other axis unresolved.
    # g_d = ln(n(lam,L_tgt) * N_other_tgt) / ln(n(lam,L_nat) * N_other_nat).
    # lam -> 0: global Jin ratio. lam >= max(L_tgt,L_nat): other-axis ratio.
    target = (length / wavelength).clamp(min=1, max=length) * other_length
    native = (native_length / wavelength).clamp(min=1, max=native_length) * native_other
    return target.log() / native.log()


class GainController:
    """Compute once per transformer call, then share gains across every block."""

    MODES = ("none", "yarn_temp", "sega", "sega_static", "constancy", "temp_only",
             "constancy_nogate", "constancy_sref", "sega_official", "sega_official_static",
             "constancy_tm", "constancy_tm_nogate", "alias_temp", "alias_temp_sref", "sega_ctm")

    def __init__(self, mode="none", grid=(64, 64), axes_dims=(16, 56, 56), rope=None,
                 profile=None, static_profile=None, eta=None, target="profile", temperature=None,
                 probe_band=None, probe_factor=1.5, gate=None):
        if gate not in (None, "on", "off"):
            raise ValueError("Gate must be on or off")
        if gate is not None:
            if mode not in ("constancy_tm", "constancy_tm_nogate"):
                raise ValueError("--gate is supported for constancy_tm; existing ablations use their named modes")
            mode = "constancy_tm_nogate" if gate == "off" else "constancy_tm"
        if mode not in self.MODES:
            raise ValueError(f"Unknown controller {mode}")
        self.mode, self.grid, self.axes_dims, self.rope = mode, tuple(grid), tuple(axes_dims), rope
        self.profile = NativeProfile.load(profile) if isinstance(profile, (str, bytes)) else profile
        if profile is not None and hasattr(profile, "__fspath__"):
            self.profile = NativeProfile.load(profile)
        self.eta, self.target = float((0.5 if mode == "sega_ctm" else 1.0) if eta is None else eta), target
        if not math.isfinite(self.eta) or self.eta < 0 or target not in ("profile", "slope"):
            raise ValueError("eta must be finite and nonnegative; target must be profile or slope")
        fixed_gain = mode in ("alias_temp", "alias_temp_sref", "sega_ctm")
        if fixed_gain and temperature not in (None, "none"):
            raise ValueError(f"{mode} defines its own gain; no extra temperature is allowed")
        self.temperature = temperature or ("none" if fixed_gain else
                                           "sref" if mode == "constancy_sref" else "logN")
        if self.temperature not in ("logN", "sref", "none"):
            raise ValueError("Temperature must be logN, sref, or none")
        self.probe_band, self.probe_factor = probe_band, probe_factor
        if probe_factor <= 0:
            raise ValueError("Probe factor must be positive")
        self.static = None
        if mode in ("sega_static", "sega_official_static"):
            import numpy as np
            if static_profile is None:
                raise ValueError(f"{mode} requires --sega-static-profile at the target resolution")
            with np.load(static_profile, allow_pickle=False) as data:
                if tuple(data["grid"]) != self.grid:
                    raise ValueError("SEGA-static profile resolution does not match")
                if mode == "sega_official_static":
                    if str(data.get("source_method", "")) != "sega_official":
                        raise ValueError("Official SEGA-static requires a sega_official gain profile")
                    self.static = torch.from_numpy(data["mean_qk_scale"].copy())
                    if self.static.shape != (sum(axes_dims),):
                        raise ValueError("SEGA-static profile dimensions do not match")
                    if (not torch.isfinite(self.static).all() or not (self.static > 0).all()
                            or not torch.equal(self.static[::2], self.static[1::2])):
                        raise ValueError("Invalid official SEGA-static q/k gains")
                else:
                    self.static = torch.from_numpy(data["mean_modulator"].copy())
                    if self.static.shape != (2, axes_dims[1] // 2) or axes_dims[1] != axes_dims[2]:
                        raise ValueError("SEGA-static profile dimensions do not match")
        if (mode.startswith("constancy") or mode == "sega_ctm") and self.profile is None:
            raise ValueError("Constancy requires --profile from native calibration")
        self.time_matched = mode in ("constancy_tm", "constancy_tm_nogate", "sega_ctm")
        if self.time_matched:
            self.profile.require_time_resolved()
            if target != "profile":
                raise ValueError(f"{mode} requires --target profile")
        self.reset()

    def reset(self):
        self.qk_scale = torch.ones(sum(self.axes_dims))
        self.log_g = torch.zeros(sum(self.axes_dims))
        self.history = []

    def frequencies(self, axis, device):
        if self.rope is not None:
            return self.rope.frequencies(axis).to(device=device)
        dim = self.axes_dims[axis]
        scale = self.grid[axis - 1] / 64
        base = 10000 * scale ** (2 * dim / (dim - 2))
        return base ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)

    def _temperature(self, axis=None):
        if self.temperature == "none":
            return 1.0
        if self.temperature == "sref":
            scale = math.sqrt(math.prod(self.grid)) / 64 if axis is None else self.grid[axis] / 64
            return scale ** 0.16  # SEGA m_ref is a q/k scale, while tau is a logit gain.
        return math.log(math.prod(self.grid)) / math.log(64 * 64)

    @torch.no_grad()
    def update(self, xt, t, x0hat=None, official_qk_scale=None):
        if xt.ndim != 3 or tuple(xt.shape[-2:]) != self.grid:
            raise ValueError("Controller expects C,H,W packed token-grid latents")
        t = float(t)
        if not 0 <= t <= 1:
            raise ValueError(f"Expected sigma in [0,1], got {t}")
        device = xt.device
        logg = torch.zeros(sum(self.axes_dims), device=device, dtype=torch.float32)
        info = {"step": len(self.history), "sigma": t, "mode": self.mode}
        scale = math.sqrt(math.prod(self.grid)) / 64
        offsets = (self.axes_dims[0], self.axes_dims[0] + self.axes_dims[1])
        if self.mode in ("alias_temp", "alias_temp_sref"):
            logg.fill_(math.log(math.log(math.prod(self.grid)) / math.log(64 * 64)))
            for a, off in enumerate(offsets):
                wavelength = 2 * math.pi / self.frequencies(a + 1, device)
                gain = alias_logit_gain(wavelength, self.grid[a], self.grid[1 - a])
                logg[off:off + self.axes_dims[a + 1]] = gain.log().repeat_interleave(2)
            if self.mode == "alias_temp_sref":
                # Include every pair, including axis 0. SEGA uses the longest-side scale.
                logg += 0.16 * math.log(max(max(self.grid) / 64, 1.0)) - logg.mean()
            info["gain_rule"] = "axial_alias_pool_log_ratio"
        elif self.mode == "yarn_temp":
            logg.fill_(2 * math.log(1 + 0.1 * math.log(scale)))
        elif self.mode == "sega_official_static":
            logg.copy_(2 * self.static.to(device).log())
        elif self.mode in ("sega", "sega_static"):
            if self.static is None:
                stats = xt.to(torch.float64 if xt.dtype == torch.float64 else torch.float32)
                mod, sigma = sega_modulator(stats.mean(0), self.frequencies(2, device),
                                            self.frequencies(1, device))
                mod = mod.flip(0)  # Public reference returns width first; logging uses height first.
                info["spectral_sigma"] = float(sigma)
            else:
                mod = self.static.to(device)
            logg.fill_(0.16 * math.log(scale))
            for a, off in enumerate(offsets):
                m = (self.grid[a] / 64) ** 0.08 * mod[a]
                logg[off:off + self.axes_dims[a + 1]] = 2 * m.abs().clamp_min(1e-30).log().repeat_interleave(2)
            info["modulator"] = mod.cpu().tolist()
        elif self.mode.startswith("constancy") or self.mode in ("temp_only", "sega_ctm"):
            if self.mode == "sega_ctm":
                if official_qk_scale is None:
                    raise ValueError("sega_ctm requires the released SEGA projected-feature gain")
                logg.copy_(2 * official_qk_scale.to(device).log())
                info["official_log_g"] = logg.cpu().tolist()
            else:
                logg.fill_(math.log(self._temperature()))
                for a, off in enumerate(offsets):
                    logg[off:off + self.axes_dims[a + 1]] = math.log(self._temperature(a))
            if self.mode != "temp_only" and x0hat is not None:
                xt_axes, x0_axes = axis_power(xt), axis_power(x0hat)
                bins = []
                for a, (et, e0, off) in enumerate(zip(xt_axes, x0_axes, offsets)):
                    rho = (et.power / (t * t * et.noise + 1e-12) - 1).clamp_min(0)
                    measured_gate = rho / (1 + rho)
                    gate = (torch.ones_like(measured_gate)
                            if self.mode in ("constancy_nogate", "constancy_tm_nogate") else measured_gate)
                    native = (self.profile.evaluate_time(a, e0.frequency, t) if self.time_matched
                              else self.profile.evaluate(a, e0.frequency, self.target))
                    loge = e0.power.clamp_min(1e-30).log()
                    anchor = (e0.frequency >= 1 / 64) & (e0.frequency <= 1 / 16)
                    if not self.time_matched:
                        anchor = anchor & (gate > 0.5)
                    deficit = torch.zeros_like(loge)
                    if anchor.any():
                        c = (loge[anchor] - native[anchor]).mean()
                        deficit = 0.5 * (native + c - loge)
                    else:
                        c = torch.zeros((), device=device)
                        if self.time_matched:
                            deficit = 0.5 * (native - loge)
                    correction = self.eta * gate * deficit.clamp(-1, 1)
                    frequencies = self.frequencies(a + 1, device) / (2 * math.pi)
                    # Interpolate across empty bands and hold the endpoint corrections fixed.
                    dim_delta = log_interpolate(frequencies, e0.frequency, correction, extrapolate=False)
                    dim_delta = torch.where(frequencies < 1 / self.grid[a], 0, dim_delta)
                    logg[off:off + self.axes_dims[a + 1]] += dim_delta.repeat_interleave(2)
                    bins.append({"axis": "height" if a == 0 else "width", "frequency": e0.frequency.cpu().tolist(),
                                 "log_power": loge.cpu().tolist(), "gate": gate.cpu().tolist(),
                                 "measured_gate": measured_gate.cpu().tolist(), "deficit": deficit.cpu().tolist(),
                                 "correction": correction.cpu().tolist(), "anchor_count": int(anchor.sum()),
                                 "anchor_offset": float(c)})
                info["bins"] = bins
            info["first_step_no_correction"] = x0hat is None
        if self.probe_band is not None:
            low, high = self.probe_band
            for a, off in enumerate(offsets):
                f = self.frequencies(a + 1, device) / (2 * math.pi)
                mask = ((f >= low) & (f < high)).repeat_interleave(2)
                logg[off:off + self.axes_dims[a + 1]] += mask * math.log(self.probe_factor)
            info["probe_band"] = list(self.probe_band)
            info["probe_factor"] = self.probe_factor
        self.log_g = logg
        self.qk_scale = (logg / 2).exp()
        self._log(info)
        return self.qk_scale

    def _log(self, info):
        total = self.log_g
        if self.rope is not None and hasattr(self.rope, "embedding_gain"):
            total = total + 2 * self.rope.embedding_gain(device=total.device).clamp_min(1e-30).log()
        info["log_g"] = total.cpu().tolist()
        info["controller_log_g"] = self.log_g.cpu().tolist()
        info["log_g_axes"] = {name: part.cpu().tolist() for name, part in
                              zip(("index", "height", "width"), total.split(self.axes_dims))}
        self.history.append(info)


def gained_attention(query, key, value, qk_scale, attention_mask=None):
    """B,S,H,D input, SDPA output, with m**2 as the per-dimension logit gain."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    m = qk_scale.to(device=query.device, dtype=query.dtype)
    # Limit the flash-only policy to FLUX blocks. CLIP/T5 can have short masked
    # attention calls that need another SDPA backend. A math fallback here would
    # materialize quadratic attention for 65,536 image tokens.
    # cuDNN attention is the fast path on Hopper (GH200); flash and memory-efficient kernels are fallbacks.
    backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    with sdpa_kernel(backends, set_priority=True) if query.is_cuda else nullcontext():
        return F.scaled_dot_product_attention((query * m).transpose(1, 2), (key * m).transpose(1, 2),
                                             value.transpose(1, 2), attn_mask=attention_mask,
                                             dropout_p=0.0, is_causal=False).transpose(1, 2)


class FluxGainAttnProcessor:
    """Adapted from diffusers FluxAttnProcessor (Apache-2.0), using PyTorch SDPA."""

    def __init__(self, controller):
        self.controller = controller

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.transformers.transformer_flux import _get_qkv_projections

        q, k, v, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if attn.added_kv_proj_dim is not None:
            if encoder_hidden_states is None:
                raise ValueError("Joint FLUX attention requires encoder_hidden_states")
            eq, ek, ev = (x.unflatten(-1, (attn.heads, -1)) for x in (eq, ek, ev))
            eq, ek = attn.norm_added_q(eq), attn.norm_added_k(ek)
            q, k, v = (torch.cat([a, b], dim=1) for a, b in ((eq, q), (ek, k), (ev, v)))
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        hidden_states = gained_attention(q, k, v, self.controller.qk_scale, attention_mask)
        hidden_states = hidden_states.flatten(2, 3).to(q.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1)
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.contiguous()))
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


def install_gain(transformer, controller):
    processors = {name: FluxGainAttnProcessor(controller) for name in transformer.attn_processors}
    if not processors:
        raise ValueError("Transformer has no attention processors")
    transformer.set_attn_processor(processors)
