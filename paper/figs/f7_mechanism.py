"""F7: correction activity and prompt-specific gain responses.

Regenerate:  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f7_mechanism.py

(a) Correction activity per denoising step from the JSON sidecars of the eight Krea dev prompts (rows 0-7, seed 0):
    late-profile target  out_krea/pilot/constancy/4096x4096/*.json          (Constancy, NTK base, log-N temperature)
    time-matched target  out_krea/devtm_eta1/constancy_tm/4096x4096/*.json  (same base and temperature, eta = 1)
    activity = mean over the 112 spatial head dims of |log g - log tau|, where log tau = ln(ln N / ln N_native)
    = ln(16/12) is the global log-N temperature; the script checks that the 16 text/time dims carry exactly log tau
    at every step. Line = median over prompts, band = interquartile range over the 8 prompts. Lower maps: median over
    prompts of log g - log tau per rotary pair, averaged over the height and width pairs.
(b) Attention-gain transfer probe, probe_krea/4096x4096{,_p1,_p2}/mtf.csv (3 prompts, seed 0, NTK base): the
    logit gain of the rotary pairs in one octave is multiplied by 1.5 on both spatial axes for the whole sampling
    run; response = d ln(final-latent band power) / d ln(gain) by same-seed forward difference. Show one
    width-axis matrix per prompt, on a common zero-centered scale. No averaging or sign-agreement overlay.
    Octave edges are in cycles per latent token; height responses remain in the printed numeric summary.
Palette: both targets are per-band laws (law orange), told apart by line style as in PALETTE.md (late-profile =
Constancy, dotted with circles; time-matched = Constancy-TM, solid with diamonds). Signed maps run method blue,
white, law orange. The grey region (steps with zero late-profile correction in every prompt) is explained in the
caption, not in the artwork.
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

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
    order = cs.prompt_order()
    ids = sorted(d.prompt_id.unique(), key=lambda pid: order[pid])
    assert [order[pid] for pid in ids] == [0, 1, 2]
    edges = np.unique(np.r_[d.output_low.values, d.output_high.values])
    mats = {}
    for axis in ("height", "width"):
        e = d[d.axis == axis]
        assert len(e) == 3 * 6 * 7 and e.response.notna().all()
        assert not e.duplicated(["prompt_id", "input_bin", "output_bin"]).any()
        per = []
        for pid in ids:
            matrix = e[e.prompt_id == pid].pivot(index="input_bin", columns="output_bin", values="response")
            assert matrix.shape == (6, 7) and np.isfinite(matrix.values).all()
            per.append(matrix.sort_index().sort_index(axis=1).values)
        mats[axis] = np.asarray(per)
    return edges, mats, [order[pid] for pid in ids]


def add_ax(fig, x, top, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H])


def frac(e):
    """Octave edge 2^-k as the plain fraction 1/2^k: no superscript, so the label prints at the tick size."""
    k = int(round(-np.log2(e)))
    return f"1/{2 ** k}" if k > 0 else str(2 ** -k)


SIGNED = LinearSegmentedColormap.from_list("signed", [cs.METHOD, "#FFFFFF", cs.LAW])   # paper-wide signed map
LINE = {"late": dict(ls=":", marker="o", ms=2.4, name="Late-profile"),      # Constancy in PALETTE.md
        "tm": dict(ls="-", marker="D", ms=2.2, name="Time-matched")}       # Constancy-TM in PALETTE.md


def minus(v):
    return f"{v:g}".replace("-", "−")


def slim_colorbar(fig, cax, norm, lim, label):
    """Vertical colorbar, 3 ticks (-lim, 0, lim) at 7 pt, math label at 8 pt."""
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SIGNED), cax=cax)
    cb.set_ticks([-lim, 0, lim])
    cb.set_ticklabels([minus(-lim), "0", minus(lim)])
    cb.ax.tick_params(labelsize=cs.FS_TICK, length=2, width=0.5, pad=1.5, colors=cs.INK2, labelcolor=cs.INK)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(cs.RULE)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.set_label(label, fontsize=cs.FS_LABEL, labelpad=3)
    return cb


def frame(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color(cs.RULE)
    ax.tick_params(length=2, width=0.5, pad=1.5, labelsize=cs.FS_TICK, colors=cs.INK2, labelcolor=cs.INK)


def main():
    cs.style()
    cs.script_floor(cs.FS_LABEL)    # the out subscript of the 8 pt colorbar label prints at 6.5 pt, not 5.6
    T = load_targets()
    edges, P, prompt_rows = load_probe()
    steps = np.arange(1, 29)

    # ---- geometry (inches from the top left)
    W = cs.TEXT
    TAG_A = 0.13
    aL, aW = 0.50, 2.10              # (a) line plot
    a_top, a_h = 0.56, 1.28
    NP = 12                          # pairs shown in the maps (pairs >= 9 are never corrected; asserted)
    hmL, hmW, hmH = 3.08, 1.72, 0.46
    hm_tops = (a_top, a_top + a_h - hmH)       # top map level with the line plot top, bottom map with its bottom
    TAG_B = a_top + a_h + 0.50
    bL, bW, b_gap = 0.62, 1.28, 0.14
    bH = bW * 6 / 7
    b_top = TAG_B + 0.22
    H = b_top + bH + 0.36
    fig = plt.figure(figsize=(W, H))

    # ---- (a) correction size per step: median and interquartile band over the eight prompts
    ax = add_ax(fig, aL, a_top, aW, a_h)
    cs.clean_axes(ax, grid="y")
    late = T["late"]["act"]
    sig = T["late"]["sigma"]
    off = np.where(late.max(0) > 0)[0][0]           # steps 1..off have zero correction in every prompt
    on_all = np.where(late.min(0) > 0)[0][0]        # first step (0-based) nonzero in every prompt
    ax.axvspan(0.5, off + 0.5, color=cs.PANEL, zorder=0, lw=0)
    med = {}
    for key, _, _, _, _ in TARGETS:
        a = T[key]["act"]
        q1, q3 = np.percentile(a, [25, 75], axis=0)
        med[key] = np.median(a, 0)
        ax.fill_between(steps, q1, q3, color=cs.LAW, alpha=0.18, lw=0, zorder=1)
        st = LINE[key]
        ax.plot(steps, med[key], color=cs.LAW, lw=1.1, ls=st["ls"], marker=st["marker"], ms=st["ms"],
                markevery=(1, 3), mec="none", zorder=3, solid_capstyle="round", dash_capstyle="round")
    ax.set_xlim(0.5, 28.5)
    ax.set_ylim(-0.004, 0.105)                     # lifts the zero-valued late-profile line off the spine
    ax.set_yticks([0, 0.05, 0.1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([1, 7, 14, 21, 28])
    ax.set_ylabel(r"Mean $|\ln g-\ln\tau|$", fontsize=cs.FS_LABEL, labelpad=3)
    ax.set_xlabel("Denoising step", fontsize=cs.FS_LABEL, labelpad=2)
    ax.xaxis.label.set_x(((aL + hmL + hmW) / 2 - aL) / aW)   # one step title, centred under the line plot and maps
    # direct labels in ink: time-matched above and left of its rising line; late-profile just above its flat
    # zero segment, ending where the grey region ends
    assert med["tm"][10] < 0.012 and np.all(med["late"][:off] == 0)
    ax.text(11.2, 0.019, LINE["tm"]["name"], fontsize=cs.FS_LABEL, color=cs.INK, ha="right", va="bottom")
    ax.text(off + 0.2, 0.0022, LINE["late"]["name"], fontsize=cs.FS_LABEL, color=cs.INK, ha="right", va="bottom")
    sec = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, steps, sig),
                                                lambda s_: np.interp(s_, sig[::-1], steps[::-1])))
    sec.set_xticks([1.0, 0.6, 0.2])
    sec.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    sec.tick_params(labelsize=cs.FS_TICK, length=2.5, width=0.6, pad=1.5, colors=cs.INK2, labelcolor=cs.INK)
    sec.spines["top"].set_linewidth(0.6)
    sec.spines["top"].set_color(cs.INK2)
    sec.set_xlabel(r"Noise level $\sigma$", fontsize=cs.FS_LABEL, labelpad=3)

    # per-pair maps (median over prompts, mean of height and width pairs), titled by target
    for k in ("late", "tm"):
        assert np.all(T[k]["pair"][:, :, 9:] == 0), "a pair beyond 8 was corrected"
    vm = np.ceil(max(np.abs(np.median(T[k]["pair"], 0)).max() for k in ("late", "tm")) * 20) / 20
    hn = Normalize(-vm, vm)
    hms = []
    for i, (key, _, _, _, _) in enumerate(TARGETS):
        hx = add_ax(fig, hmL, hm_tops[i], hmW, hmH)
        hx.sharex(ax)
        if hms:
            hx.sharey(hms[0])
        hms.append(hx)
        m = np.median(T[key]["pair"], 0)[:, :NP]          # (28 steps, NP pairs)
        hx.pcolormesh(np.arange(0.5, 29), np.arange(-0.5, NP), m.T, cmap=SIGNED, norm=hn, shading="flat",
                      edgecolors="face", linewidth=0.25)
        hx.set_ylim(NP - 0.5, -0.5)
        hx.set_yticks([0, NP - 1])
        frame(hx)
        hx.text(0.0, 1.0, LINE[key]["name"], transform=hx.transAxes, fontsize=cs.FS_LABEL, color=cs.INK,
                ha="left", va="bottom")
        hx.set_title("")
        if i == 0:
            hx.tick_params(labelbottom=False)
    hms[0].set_ylabel("Pair", fontsize=cs.FS_LABEL, labelpad=2)
    hms[0].yaxis.label.set_y(1 - a_h / (2 * hmH))    # centred on both maps
    for t in hms[0].texts + hms[1].texts:            # titles sit 2 pt above their map
        t.set_transform(t.get_transform() + plt.matplotlib.transforms.ScaledTranslation(0, 2 / 72,
                                                                                       fig.dpi_scale_trans))
    hcb = add_ax(fig, hmL + hmW + 0.07, hm_tops[0], 0.08, a_h)
    slim_colorbar(fig, hcb, hn, vm, r"$\ln g-\ln\tau$")

    # ---- (b) prompt-specific width-axis transfer, no cross-prompt average
    lg = np.log2(edges)
    vb = np.ceil(np.abs(P["width"]).max() * 10) / 10
    bn = Normalize(-vb, vb)
    matrix_axes = []
    for i, (prompt_row, matrix) in enumerate(zip(prompt_rows, P["width"])):
        mx = add_ax(fig, bL + i * (bW + b_gap), b_top, bW, bH)
        if matrix_axes:
            mx.sharey(matrix_axes[0])
            mx.sharex(matrix_axes[0])
        matrix_axes.append(mx)
        mx.pcolormesh(lg, lg[:7], matrix, cmap=SIGNED, norm=bn, shading="flat",
                      edgecolors="face", linewidth=0.25, rasterized=False)
        mx.set_xlim(lg[0], lg[-1])
        mx.set_ylim(lg[0], lg[6])
        mx.set_aspect("equal")
        # two output labels per matrix, six along the shared axis: a 1/256 at every left edge ran into the
        # previous matrix's 1/4; the left edge is the y axis's 1/256 (both axes start at the same octave edge)
        mx.set_xticks(lg[[3, 6]])
        mx.set_xticklabels([frac(edges[j]) for j in (3, 6)])
        mx.set_yticks(lg[[0, 3, 6]])
        mx.set_yticklabels([frac(edges[j]) for j in (0, 3, 6)])
        frame(mx)
        mx.tick_params(axis="x", pad=3.5)              # unchanged, so the output-axis title keeps its place
        if i:
            mx.tick_params(labelleft=False)
        mx.text(0.5, 1.0, f"Prompt {prompt_row}", transform=mx.transAxes, fontsize=cs.FS_LABEL, color=cs.INK,
                ha="center", va="bottom")
    matrix_axes[0].set_ylabel("Gained octave", fontsize=cs.FS_LABEL, labelpad=2)
    matrix_axes[1].set_xlabel("Output octave (cycles/token)", fontsize=cs.FS_LABEL, labelpad=2)
    for mx in matrix_axes:
        for t in mx.texts:
            t.set_transform(t.get_transform() + plt.matplotlib.transforms.ScaledTranslation(0, 3 / 72,
                                                                                           fig.dpi_scale_trans))
    bcb = add_ax(fig, bL + 3 * bW + 2 * b_gap + 0.08, b_top, 0.08, bH)
    slim_colorbar(fig, bcb, bn, vb, r"$\Delta\ln P_{\mathrm{out}}/\ln 1.5$")

    cs.panel_tag(fig, 0.0, 1 - TAG_A / H, "a", "Correction size")
    cs.panel_tag(fig, 0.0, 1 - TAG_B / H, "b", "Band response")
    print(f"figure size {W:.2f} x {H:.3f} in")
    cs.finish(fig, str(cs.HERE / "f7_mechanism"), photo=False)

    print(f"late-profile: zero through step {off} (sigma {sig[off - 1]:.3f}); first nonzero step {off + 1} "
          f"(sigma {sig[off]:.3f}); nonzero in all prompts from step {on_all + 1} (sigma {sig[on_all]:.3f})")
    tm = T["tm"]["act"]
    print("time-matched: first step nonzero in all prompts", np.where(tm.min(0) > 0)[0][0] + 1)
    for k in ("late", "tm"):
        print(k, "median activity at steps 10/20/28:", np.round(np.median(T[k]["act"], 0)[[9, 19, 27]], 4))
    print("map scale +-", vm, "; probe scale +-", vb)
    for axis in P:
        for prompt_row, matrix in zip(prompt_rows, P[axis]):
            print(axis, "dev prompt", prompt_row, "diagonal", np.round(np.diag(matrix[:, :6]), 3))


if __name__ == "__main__":
    main()
