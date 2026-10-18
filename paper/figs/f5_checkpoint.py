"""F5: checkpoint dependence. FLUX.1-dev breaks into a dot lattice at 4096^2, FLUX.1-Krea-dev does not.

Regenerate:  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f5_checkpoint.py

Inputs (scratch, read only):
  diag/official_dype_{dev_4096_g3.5,krea_4096_g4.5}_dype1.png   official DyPE code, prompt 0, seed 42
  out/pilot/{dype,sega_official}/4096x4096/<prompt 0>_*.png      our pipeline on FLUX.1-dev, seed 0
  out_krea/pilot/{dype,sega_official}/4096x4096/<prompt 0>_*.png our pipeline on FLUX.1-Krea-dev, seed 0
Prompt 0 is the first row of prompts/aesthetic4k_eval.jsonl (the official script reads that row).
Crop rule: two fixed 192 px windows, identical in every image, centred on the vertical midline at one quarter
(A) and three quarters (B) of the frame height. Crops are native pixels; full frames are Lanczos reductions.
Spectrum (b): radially averaged power of the full 4096^2 luminance (Hann window), times f^2, for the image pair
shown above each plot. Peak ratio = dev / Krea whitened power at the dev peak (periods 8 to 64 px); for our pipeline
it is the geometric mean over the 8 dev prompts (rows 0-7, seed 0) with a 95% bootstrap CI (2,000 resamples).
"""
import glob
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator
from PIL import Image

import crope_style as cs

S = cs.SCR
PID = "a4k_eval_4096_minus6873203881450547484"
CROP_PX = 192
CENTERS = {"A": (2048, 1024), "B": (2048, 3072)}
CK_INK = {"dev": cs.INK, "krea": "#A3A9B1"}  # checkpoint lines: dark = FLUX.1-dev, light = Krea
TAG_UP = 0.13         # panel-letter baseline above the top of panel b's plots (in)
SPEC_IN = 0.15        # spectrum plots start this far inside their image column (room for log tick labels)


def sources():
    first = json.loads(open(S / "prompts" / "aesthetic4k_eval.jsonl").readline())
    assert first["id"] == PID, "official DyPE diag prompt is not the pilot prompt 0"
    cols = []
    cols.append(dict(key="official", method="dype", line2="official code, seed 42",
                     dev=S / "diag/official_dype_dev_4096_g3.5_dype1.png",
                     krea=S / "diag/official_dype_krea_4096_g4.5_dype1.png"))
    for m in ("dype", "sega_official"):
        col = dict(key=m, method=m, line2="our pipeline, seed 0")
        for ck, root, g in (("dev", "out", 3.5), ("krea", "out_krea", 4.5)):
            png = glob.glob(str(S / root / "pilot" / m / "4096x4096" / f"{PID}_*.png"))
            assert len(png) == 1, (root, m)
            side = json.load(open(png[0][:-4] + ".json"))
            c = side["config"]
            assert c["method"] == m and c["prompt_id"] == PID and c["prompt"] == first["prompt"]
            assert c["seed"] == 0 and c["guidance"] == g and c["res"] == "4096x4096" and side["complete"]
            col[ck] = png[0]
            # all eight dev prompts (rows 0-7) of this method and checkpoint, for the spectrum statistics
            every = sorted(glob.glob(str(S / root / "pilot" / m / "4096x4096" / "*.png")))
            assert len(every) == 8 and all(cs.prompt_order()[json.load(open(q[:-4] + ".json"))["config"]
                                                               ["prompt_id"]] < 8 for q in every)
            col[ck + "_all"] = every
        assert [q.split("/")[-1] for q in col["dev_all"]] == [q.split("/")[-1] for q in col["krea_all"]]
        cols.append(col)
    return first["prompt"], cols


def box(center):
    cx, cy = center
    h = CROP_PX // 2
    return (cx - h, cy - h, cx + h, cy + h)


def whitened_spectrum(path):
    """f^2 * radially averaged power of the Hann-windowed full-frame luminance (cached on scratch)."""
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime_ns}|radial-v1".encode()).hexdigest()[:20]
    out = cs.CACHE / f"{key}.npz"
    if out.exists():
        z = np.load(out)
        return z["f"], z["p"]
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), np.float64) / 255.0
    y = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    n = y.shape[0]
    w = np.hanning(n)
    y = (y - y.mean()) * w[:, None] * w[None, :]
    P = np.abs(np.fft.fft2(y)) ** 2 / (n * n)
    f = np.fft.fftfreq(n)
    R = np.hypot(f[:, None], f[None, :]).ravel()
    edges = np.geomspace(2.0 / n, 0.5, 49)
    idx = np.digitize(R, edges)
    pw = np.bincount(idx, weights=P.ravel(), minlength=len(edges) + 1)[1:len(edges)]
    ct = np.bincount(idx, minlength=len(edges) + 1)[1:len(edges)]
    fc = np.sqrt(edges[1:] * edges[:-1])
    p = pw / np.maximum(ct, 1) * fc ** 2
    cs.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, f=fc, p=p)
    return fc, p


def peak_ratio(dev_png, krea_png):
    """dev / Krea whitened power at the dev peak, searched over periods of 8 to 64 px."""
    f, pd_ = whitened_spectrum(dev_png)
    _, pk = whitened_spectrum(krea_png)
    i = np.argmax(np.where((f > 1 / 64) & (f < 1 / 8), pd_, -np.inf))
    return pd_[i] / pk[i]


def add_ax(fig, x, top, w, h):
    """Axes at x (in from left) and top (in from top), size w x h inches."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H])


def photo(ax, arr, hairline=False):
    ax.imshow(arr, interpolation="none")
    ax.set_axis_off()
    if hairline:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, lw=0.4,
                               edgecolor=cs.RULE, clip_on=False))


HALO = [pe.withStroke(linewidth=1.8, foreground=cs.INK)]
TXT_HALO = [pe.withStroke(linewidth=1.4, foreground=cs.INK)]


def main():
    cs.style()
    prompt, cols = sources()

    # ---- geometry (inches)
    W = cs.TEXT
    lab_w = 0.24          # row-label strip
    gut = 0.12            # column gutter
    gap = 0.03            # frame-to-crop gap
    F = (W - lab_w - 2 * gut - 1.5 * gap) / 4.5   # full-frame side: 3 * (F + gap + C) + gutters = W
    C = (F - gap) / 2     # crop side
    colw = F + gap + C
    head = 0.40           # column headers
    rowgap = 0.07
    spec_top = head + 2 * F + rowgap + 0.42
    spec_h = 0.84
    H = spec_top + spec_h + 0.36
    fig = plt.figure(figsize=(W, H))

    rows = (("dev", "FLUX.1-dev", "guidance 3.5"), ("krea", "FLUX.1-Krea-dev", "guidance 4.5"))
    for ci, col in enumerate(cols):
        x0 = lab_w + ci * (colw + gut)
        # header: method name in its color over a rule of the same color, provenance below in secondary ink
        m = col["method"]
        xc = (x0 + colw / 2) / W
        fig.text(xc, 1 - 0.14 / H, cs.label(m), fontsize=7.5, color=cs.text_color(m), ha="center", va="baseline",
                 fontweight="bold")
        fig.add_artist(plt.Line2D([x0 / W, (x0 + colw) / W], [1 - 0.185 / H] * 2, lw=0.8, color=cs.color(m),
                                  transform=fig.transFigure, solid_capstyle="butt"))
        fig.text(xc, 1 - 0.31 / H, col["line2"], fontsize=6.5, color=cs.INK2, ha="center", va="baseline")
        for ri, (ck, name, g) in enumerate(rows):
            top = head + ri * (F + rowgap)
            ax = add_ax(fig, x0, top, F, F)
            photo(ax, cs.thumb(col[ck], 768))
            s = 768 / 4096
            for tag, c in CENTERS.items():
                b = box(c)
                r = Rectangle((b[0] * s, b[1] * s), CROP_PX * s, CROP_PX * s, fill=False, lw=0.6,
                              edgecolor="white")
                r.set_path_effects(HALO)
                ax.add_patch(r)
                t = ax.text(b[2] * s + 10, b[1] * s + CROP_PX * s / 2, tag, color="white", fontsize=6,
                            ha="left", va="center", fontweight="bold")
                t.set_path_effects(TXT_HALO)
            for k, tag in enumerate(CENTERS):
                cax = add_ax(fig, x0 + F + gap, top + k * (C + gap), C, C)
                photo(cax, cs.crop(col[ck], box(CENTERS[tag])), hairline=True)
                t = cax.text(0.06, 0.94, tag, transform=cax.transAxes, color="white", fontsize=6,
                             ha="left", va="top", fontweight="bold")
                t.set_path_effects(TXT_HALO)
            if ci == 0:
                fig.text(0.07 / W, 1 - (top + F / 2) / H, name, rotation=90, fontsize=7.5, color=cs.INK,
                         ha="center", va="center", fontweight="bold")
                fig.text(0.19 / W, 1 - (top + F / 2) / H, g, rotation=90, fontsize=6.5, color=cs.INK2,
                         ha="center", va="center")

    # ---- spectra row, one plot per column, shared axes
    axs = []
    for ci, col in enumerate(cols):
        x0 = lab_w + ci * (colw + gut)
        ax = add_ax(fig, x0 + SPEC_IN, spec_top, colw - SPEC_IN, spec_h)
        if axs:
            ax.sharex(axs[0])
            ax.sharey(axs[0])
        axs.append(ax)
        peak = None
        for ck in ("krea", "dev"):
            f, p = whitened_spectrum(col[ck])
            ok = f > 2.5e-3
            ax.plot(f[ok], p[ok], color=CK_INK[ck], lw=1.1 if ck == "dev" else 1.0, solid_capstyle="round",
                    zorder=3 if ck == "dev" else 2)
            if ck == "dev":
                band = (f > 1 / 128) & (f < 1 / 4)
                i = np.argmax(np.where(band, p, -np.inf))
                peak = (f[i], p[i])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([peak[0]], [peak[1]], marker="o", ms=2.6, color=cs.INK, zorder=4)
        ax.annotate(f"period \u2248 {1 / peak[0]:.0f} px", xy=peak, xytext=(0, 4), textcoords="offset points",
                    fontsize=6.5, color=cs.INK, ha="center", va="bottom")
        if col["key"] == "official":
            r = peak_ratio(col["dev"], col["krea"])
            txt = f"peak ratio {r:.0f}\u00d7 (this pair only)"
        else:
            r = np.log([peak_ratio(a, b) for a, b in zip(col["dev_all"], col["krea_all"])])
            gm, lo, hi = np.exp(cs.boot(r))
            txt = f"8 prompts: peak ratio {gm:.0f}\u00d7 [{lo:.0f}, {hi:.0f}]"
            print(col["key"], "per-prompt peak ratios", np.round(np.exp(r), 1))
        ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize=6, color=cs.INK2, ha="left", va="bottom")
        ax.grid(True, which="major", axis="both", lw=0.4, color=cs.GRIDC)
        ax.tick_params(labelsize=7)
        if ci > 0:
            ax.tick_params(labelleft=False)
    axs[0].set_xlim(2.5e-3, 0.5)
    axs[0].set_ylim(5e-8, 6e-3)
    axs[0].yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axs[0].yaxis.set_minor_formatter(NullFormatter())
    # y label set horizontally above the first plot (tick labels fill the strip to its left)
    axs[0].set_ylabel(r"Whitened power $f^{2}P(f)$, a.u.", fontsize=7, rotation=0, ha="left", va="baseline")
    axs[0].yaxis.set_label_coords((0.16 - (lab_w + SPEC_IN)) / (colw - SPEC_IN), 1 + TAG_UP / spec_h)
    axs[0].yaxis.set_minor_locator(NullLocator())
    axs[1].set_xlabel("Spatial frequency $f$ (cycles per pixel)", fontsize=7.5)
    # checkpoint key on the label line above the plots (the curves cross, so a key reads cleaner)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CK_INK[k], lw=1.1) for k in ("dev", "krea")]
    fig.legend(handles, ["FLUX.1-dev", "FLUX.1-Krea-dev"], loc="lower right", ncol=2, frameon=False,
               fontsize=6.5, handlelength=1.6, columnspacing=1.2, handletextpad=0.5, borderaxespad=0,
               bbox_to_anchor=(1.0, 1 - (spec_top - TAG_UP + 0.035) / H))

    cs.panel_tag(fig, 0.0, 1 - 0.14 / H, "a")
    cs.panel_tag(fig, 0.0, 1 - (spec_top - TAG_UP) / H, "b")
    cs.finish(fig, str(cs.HERE / "f5_checkpoint"), photo=True)
    print("prompt:", prompt)


if __name__ == "__main__":
    main()
