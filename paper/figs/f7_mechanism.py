"""F7: why the per-band controllers barely move the image: when the correction acts, and how a gain transfers.

Regenerate:  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f7_mechanism.py

(a) Correction activity per denoising step from the JSON sidecars of the eight Krea dev prompts (rows 0-7, seed 0):
    late-profile target  out_krea/pilot/constancy/4096x4096/*.json          (Constancy, NTK base, log-N temperature)
    time-matched target  out_krea/devtm_eta1/constancy_tm/4096x4096/*.json  (same base and temperature, eta = 1)
    activity = mean over the 112 spatial head dims of |log g - log tau|, where log tau = ln(ln N / ln N_native)
    = ln(16/12) is the global log-N temperature; the script checks that the 16 text/time dims carry exactly log tau
    at every step. Line = median over prompts, band = full range over the 8 prompts. Lower maps: median over
    prompts of log g - log tau per rotary pair, averaged over the height and width pairs.
(b) Attention-gain transfer probe, probe_krea/4096x4096{,_p1,_p2}/mtf.csv (3 prompts, seed 0, NTK base): the
    logit gain of the rotary pairs in one octave is multiplied by 1.5 on both spatial axes for the whole sampling
    run; response = d ln(final-latent band power) / d ln(gain) by same-seed forward difference, mean over the
    3 prompts, for power measured along height and along width. Octave edges in cycles per latent token.
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import crope_style as cs

S = cs.SCR
TARGETS = [("late", "pilot/constancy", "constancy", cs.SAFFRON, "Late-profile target (Constancy)"),
           ("tm", "devtm_eta1/constancy_tm", "constancy_tm", cs.TERRA, "Time-matched target")]
PROBES = ["4096x4096", "4096x4096_p1", "4096x4096_p2"]


def load_targets():
    order = cs.prompt_order()
    out = {}
    for key, root, method, _, _ in TARGETS:
        files = sorted(glob.glob(str(S / "out_krea" / root / "4096x4096" / "*.json")))
        assert len(files) == 8, (root, len(files))
        lg, sig, ids = [], None, []
        for f in files:
            j = json.load(open(f))
            c = j["config"]
            assert c["method"] == method and c["temperature"] == "logN" and c["rope"]["method"] == "ntk2"
            assert c["eta"] == 1.0 and c["seed"] == 0 and order[c["prompt_id"]] < 8 and c["res"] == "4096x4096"
            n_tok = np.prod(c["rope"]["grid_tokens"])
            n_nat = np.prod(c["rope"]["native_grid_tokens"])
            log_tau = np.log(np.log(n_tok) / np.log(n_nat))          # log-N temperature, ln(16/12) at 4096^2
            a = np.array([g["log_g"] for g in j["gains"]])
            s = np.array([g["sigma"] for g in j["gains"]])
            assert a.shape == (28, 128) and np.allclose(a[:, :16], log_tau, atol=1e-6), "temperature term"
            assert np.all(a[:, :16] == a[:, :1])
            assert np.allclose(a[:, 0::2], a[:, 1::2])
            sig = s if sig is None else sig
            assert np.allclose(sig, s)
            lg.append(a)
            ids.append(c["prompt_id"])
        lg = np.array(lg)
        d = lg[:, :, 16:] - lg[:, :, :1]       # (8, 28, 112): spatial dims minus the stored temperature term
        out[key] = dict(act=np.abs(d).mean(2), pair=0.5 * (d[:, :, 0::2][:, :, :28] + d[:, :, 0::2][:, :, 28:]),
                        sigma=sig, ids=ids)
    assert set(out["late"]["ids"]) == set(out["tm"]["ids"]) and np.allclose(out["late"]["sigma"], out["tm"]["sigma"])
    return out


def load_probe():
    d = pd.concat([pd.read_csv(S / "probe_krea" / p / "mtf.csv") for p in PROBES], ignore_index=True)
    assert d.prompt_id.nunique() == 3 and (d.seed == 0).all() and (d.gain_factor == 1.5).all()
    d = d[d.domain == "latent"]
    edges = np.unique(np.r_[d.output_low.values, d.output_high.values])
    mats = {}
    for ax in ("height", "width"):
        e = d[d.axis == ax]
        assert len(e) == 3 * 6 * 7 and e.response.notna().all()
        mean = e.groupby(["input_bin", "output_bin"]).response.mean().unstack().values      # (6, 7)
        per = e.pivot_table(index=["prompt_id", "input_bin"], columns="output_bin", values="response")
        per = per.values.reshape(3, 6, 7)
        agree = (np.sign(per) == np.sign(mean)[None]).all(0)
        mats[ax] = (mean, agree)
    return edges, mats


def add_ax(fig, x, top, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H])


def frac(e):
    return rf"$2^{{{int(round(np.log2(e)))}}}$"


def main():
    cs.style()
    T = load_targets()
    edges, P = load_probe()
    steps = np.arange(1, 29)

    # ---- geometry (inches)
    W = cs.TEXT
    top0 = 0.42                      # panel letters and the sigma axis above panel a
    aL, aW = 0.44, 2.11              # panel a plot area
    lineH = 1.34
    NP = 12                          # pairs shown in the maps (pairs >= 9 are never corrected; asserted)
    hmH, hmGap = 0.39, 0.06
    hm_top = top0 + lineH + 0.08
    cell = 0.20
    bW, bH = 7 * cell, 6 * cell
    bL = W - 0.62 - bW                # panel b matrix left; colorbar block on its right
    b_top = top0 - 0.02
    b_gap = 0.22
    H = max(hm_top + 2 * hmH + hmGap + 0.36, b_top + 2 * bH + b_gap + 0.50)
    fig = plt.figure(figsize=(W, H))

    # ---- (a) activity
    ax = add_ax(fig, aL, top0, aW, lineH)
    late = T["late"]["act"]
    sig = T["late"]["sigma"]
    off = np.where(late.max(0) > 0)[0][0]           # steps 1..off have zero correction in every prompt
    on_all = np.where(late.min(0) > 0)[0][0]        # first step (0-based) nonzero in every prompt
    ax.axvspan(0.5, off + 0.5, color=cs.PANEL, zorder=0, lw=0)
    for key, _, _, col, name in TARGETS:
        a = T[key]["act"]
        q1, q3 = np.percentile(a, [25, 75], axis=0)
        ax.fill_between(steps, q1, q3, color=col, alpha=0.25, lw=0, zorder=1)
        ax.plot(steps, np.median(a, 0), color=col, lw=1.4, zorder=3, solid_capstyle="round")
    ax.text(1.3, 0.1, f"Late-profile correction is exactly 0 in all\n8 prompts for steps 1 to {off} "
            rf"($\sigma\geq{sig[off - 1]:.2f}$)",
            fontsize=6.5, color=cs.INK2, ha="left", va="top", linespacing=1.3)
    ax.set_xlim(0.5, 28.5)
    ax.set_ylim(0, 0.105)
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_xticks([1, 7, 14, 21, 28])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"Mean $|\log g-\log\tau|$", fontsize=7.5)
    ax.grid(True, axis="y", lw=0.4, color=cs.GRIDC)
    med_late, med_tm = np.median(late, 0), np.median(T["tm"]["act"], 0)
    ax.text(9.0, med_tm[8] + 0.011, "Time-matched", color=cs.text_color("constancy_tm_sbase"), fontsize=6.8, ha="center",
            va="bottom", fontweight="bold")
    ax.annotate("Late-profile\n(Constancy)", xy=(on_all + 0.9, med_late[on_all] + 0.002), xytext=(21.2, 0.062),
                color=cs.text_color("constancy"), fontsize=6.8, ha="right", va="center", fontweight="bold", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=cs.text_color("constancy"), shrinkA=1, shrinkB=1))
    sec = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, steps, sig),
                                                lambda s_: np.interp(s_, sig[::-1], steps[::-1])))
    sec.set_xticks([1.0, 0.8, 0.6, 0.4, 0.2])
    sec.tick_params(labelsize=6.5, length=2, pad=1)
    sec.set_xlabel(r"Noise level $\sigma$", fontsize=7, labelpad=2)

    # per-pair maps (median over prompts, mean of height and width pairs)
    for k in ("late", "tm"):
        assert np.all(T[k]["pair"][:, :, 9:] == 0), "a pair beyond 8 was corrected"
    vm = np.ceil(max(np.abs(np.median(T[k]["pair"], 0)).max() for k in ("late", "tm")) * 20) / 20
    hn = Normalize(-vm, vm)
    hms = []
    for i, (key, _, _, col, name) in enumerate(TARGETS):
        hx = add_ax(fig, aL, hm_top + i * (hmH + hmGap), aW, hmH)
        hx.sharex(ax)
        if hms:
            hx.sharey(hms[0])
        hms.append(hx)
        m = np.median(T[key]["pair"], 0)[:, :NP]          # (28 steps, NP pairs)
        hx.pcolormesh(np.arange(0.5, 29), np.arange(-0.5, NP), m.T, cmap="RdBu_r", norm=hn, shading="flat",
                      edgecolors="face", linewidth=0.25)
        hx.set_ylim(NP - 0.5, -0.5)
        hx.set_yticks([0, NP - 1])
        hx.tick_params(length=2, pad=1.5, labelsize=6.5)
        for sp in hx.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color(cs.RULE)
        hx.text(0.012, 0.5, "Late-profile" if key == "late" else "Time-matched", transform=hx.transAxes,
                fontsize=6.5, color=cs.text_color("constancy") if key == "late" else cs.text_color("constancy_tm_sbase"), va="center", ha="left", fontweight="bold")
        if i == 0:
            hx.tick_params(labelbottom=False)
    hms[1].set_xlabel("Denoising step", fontsize=7.5)
    hms[0].set_ylabel("Pair", fontsize=7, labelpad=1)
    hcb = add_ax(fig, aL + aW + 0.06, hm_top, 0.06, 2 * hmH + hmGap)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=hn, cmap="RdBu_r"), cax=hcb)
    cb.set_ticks([-vm, 0, vm])
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.set_label(r"$\log g-\log\tau$", fontsize=6.5, labelpad=1)

    # ---- (b) transfer matrices
    lg = np.log2(edges)
    vb = np.ceil(max(np.abs(P[k][0]).max() for k in P) * 10) / 10
    bn = Normalize(-vb, vb)
    mxs = []
    for i, axis in enumerate(("height", "width")):
        mx = add_ax(fig, bL, b_top + i * (bH + b_gap), bW, bH)
        if mxs:
            mx.sharex(mxs[0])
        mxs.append(mx)
        mean, agree = P[axis]
        mx.pcolormesh(lg, lg[:7], mean, cmap="RdBu_r", norm=bn, shading="flat", edgecolors="face", linewidth=0.25)
        for r in range(6):
            for c in range(7):
                cx, cy = (lg[c] + lg[c + 1]) / 2, (lg[r] + lg[r + 1]) / 2
                v = mean[r, c]
                if r == c:
                    mx.add_patch(Rectangle((lg[c], lg[r]), 1, 1, fill=False, lw=0.7, edgecolor=cs.INK, zorder=3))
                    t = "0.00" if abs(v) < 0.005 else f"{v:.2f}".replace("-", "−")
                    mx.text(cx, cy, t, fontsize=5.6, ha="center", va="center",
                            color="white" if abs(v) > 0.6 * vb else cs.INK, zorder=4)
                elif agree[r, c]:
                    mx.plot(cx, cy, marker="o", ms=1.5, color=cs.INK, alpha=0.55, mew=0, zorder=4)
        mx.set_xlim(lg[0], lg[-1])
        mx.set_ylim(lg[0], lg[6])
        mx.set_xticks(lg)
        mx.set_xticklabels([frac(e) for e in edges])
        mx.set_yticks(lg[:7])
        mx.set_yticklabels([frac(e) if (k or i == 0) else "" for k, e in enumerate(edges[:7])])  # corner clash
        mx.tick_params(length=2, pad=1.5, labelsize=6.3)
        for sp in mx.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color(cs.INK2)
        mx.set_ylabel("Gained octave", fontsize=7, labelpad=1)
        mx.text(0.0, 1.035, f"Power measured along {axis}", transform=mx.transAxes, fontsize=6.5, color=cs.INK,
                ha="left", va="bottom")
        if i == 0:
            mx.tick_params(labelbottom=False)
    mxs[1].set_xlabel("Output octave (cycles per token)", fontsize=7, labelpad=2)
    kx = bL - 0.36
    ky = b_top + 2 * bH + b_gap + 0.43
    fig.add_artist(Rectangle((kx / W, 1 - (ky + 0.005) / H), 0.075 / W, 0.075 / H, transform=fig.transFigure,
                             fill=False, lw=0.7, edgecolor=cs.INK))
    fig.text((kx + 0.11) / W, 1 - ky / H, "same octave", fontsize=6, color=cs.INK2, ha="left", va="baseline")
    fig.add_artist(plt.Line2D([(kx + 0.86) / W], [1 - (ky - 0.03) / H], marker="o", ms=1.8, color=cs.INK,
                              alpha=0.55, mew=0, transform=fig.transFigure))
    fig.text((kx + 0.93) / W, 1 - ky / H, "same sign in all 3 prompts", fontsize=6, color=cs.INK2,
             ha="left", va="baseline")
    bcb = add_ax(fig, bL + bW + 0.07, b_top, 0.06, 2 * bH + b_gap)
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=bn, cmap="RdBu_r"), cax=bcb)
    cb2.ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)
    cb2.outline.set_linewidth(0.4)
    cb2.solids.set_rasterized(False)
    cb2.solids.set_edgecolor("face")
    cb2.set_label(r"Response, $\partial\ln P_{\mathrm{out}}/\partial\ln g$", fontsize=7, labelpad=1)

    cs.panel_tag(fig, 0.0, 1 - 0.14 / H, "a")
    cs.panel_tag(fig, (bL - 0.40) / W, 1 - 0.14 / H, "b")
    cs.finish(fig, str(cs.HERE / "f7_mechanism"), photo=False)

    print(f"late-profile: zero through step {off} (sigma {sig[off - 1]:.3f}); first nonzero step {off + 1} "
          f"(sigma {sig[off]:.3f}); nonzero in all prompts from step {on_all + 1} (sigma {sig[on_all]:.3f})")
    tm = T["tm"]["act"]
    print("time-matched: first step nonzero in all prompts", np.where(tm.min(0) > 0)[0][0] + 1)
    for k in ("late", "tm"):
        print(k, "median activity at steps 10/20/28:", np.round(np.median(T[k]["act"], 0)[[9, 19, 27]], 4))
    for axis in P:
        print(axis, "diag", np.round(np.diag(P[axis][0][:, :6]), 2), "sign-agree cells", int(P[axis][1].sum()))


if __name__ == "__main__":
    main()
