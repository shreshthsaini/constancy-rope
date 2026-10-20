"""F5: checkpoint dependence. FLUX.1-dev breaks into a dot lattice at 4096^2, FLUX.1-Krea-dev does not.

Regenerate:  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f5_checkpoint.py

Inputs (scratch, read only):
  diag/official_dype_{dev_4096_g3.5,krea_4096_g4.5}_dype1.png   official DyPE code, prompt 0, seed 42
  out/pilot/{dype,sega_official}/4096x4096/<prompt 0>_*.png      our pipeline on FLUX.1-dev, seed 0
  out_krea/pilot/{dype,sega_official}/4096x4096/<prompt 0>_*.png our pipeline on FLUX.1-Krea-dev, seed 0
Prompt 0 is the first row of prompts/aesthetic4k_eval.jsonl (the official script reads that row).
Crop rule: two fixed 192 px windows, identical in every image, centred on the vertical midline at one quarter
(A, red box and frame) and three quarters (B, cyan) of the frame height. Crops are native pixels; full frames are
Lanczos reductions.
Layout: one row per pipeline (official DyPE, our DyPE, our SEGA (released)), labelled at the left; columns
FLUX.1-dev, FLUX.1-Krea-dev (labelled under the bottom row), then that row's spectra (b). Every label is plain
text: y ticks in dB (10 log10 of the a.u. power), x ticks decimal. Peak ratios are printed for the caption only.
Spectrum (b): radially averaged power of the full 4096^2 luminance (Hann window), times f^2, for the image pair
in the same row. Peak ratio = dev / Krea whitened power at the dev peak (periods 8 to 64 px); for our pipeline
it is the geometric mean over the 8 dev prompts (rows 0-7, seed 0) with a 95% bootstrap CI (2,000 resamples).
"""
import glob
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from PIL import Image

import crope_style as cs

S = cs.SCR
PID = "a4k_eval_4096_minus6873203881450547484"
CROP_PX = 192
CENTERS = {"A": (2048, 1024), "B": (2048, 3072)}
CK_INK = {"dev": cs.INK, "krea": "#A3A9B1"}  # checkpoint lines: dark = FLUX.1-dev, light = Krea


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


def photo(ax, arr):
    ax.imshow(arr, interpolation="none")
    ax.set_axis_off()


WINDOW_COLOR = {"A": cs.ZOOM, "B": cs.ZOOM2}   # window A: red box and frame; window B: cyan
CK_INK = {"dev": cs.INK, "krea": cs.GRAY}       # spectra: black = FLUX.1-dev, gray = FLUX.1-Krea-dev
ROW_LABEL = {"official": "DyPE, official", "dype": "DyPE", "sega_official": "SEGA (released)"}


def main():
    cs.style()
    prompt, cols = sources()

    # ==== geometry (inches). One row per pipeline: FLUX.1-dev cell, FLUX.1-Krea-dev cell, then that pair's spectra
    W, H = cs.TEXT, 3.26
    TAG_Y = 0.125          # panel-tag baseline, from the top
    IMG_TOP = 0.20         # top of the first image row
    STRIP = 0.18           # rotated row labels
    rowgap = 0.08
    bottom = 0.34          # column labels under the images; tick labels and x title under the spectra
    F = (H - IMG_TOP - bottom - 2 * rowgap) / 3     # full-frame side
    gap = 0.03             # frame to crops, and between the two crops
    C = (F - gap) / 2      # crop side
    cell = F + gap + C
    x_ck = {"dev": STRIP, "krea": STRIP + cell + 0.10}
    x_b = x_ck["krea"] + cell + 0.14                # panel b tag; its y title and tick labels fill the strip to spec_x
    spec_x = x_b + 0.36
    spec_w = W - spec_x - 0.03
    fig = plt.figure(figsize=(W, H))

    s = 768 / 4096
    for ri, col in enumerate(cols):
        top = IMG_TOP + ri * (F + rowgap)
        fig.text(0.075 / W, 1 - (top + F / 2) / H, ROW_LABEL[col["key"]], rotation=90, fontsize=cs.FS_LABEL,
                 color=cs.INK, ha="center", va="center",
                 fontweight="bold" if col["key"] == "sega_official" else "normal")
        for ck in ("dev", "krea"):
            x0 = x_ck[ck]
            ax = add_ax(fig, x0, top, F, F)
            photo(ax, cs.thumb(col[ck], 768))
            for k, (tag, c) in enumerate(CENTERS.items()):
                b = box(c)
                # imshow pixel i spans [i - 0.5, i + 0.5], so the window edge sits half a pixel left of its index
                cs.crop_box(ax, (b[0] * s - 0.5, b[1] * s - 0.5, CROP_PX * s, CROP_PX * s), color=WINDOW_COLOR[tag])
                cax = add_ax(fig, x0 + F + gap, top + k * (C + gap), C, C)
                photo(cax, cs.crop(col[ck], b))
                cs.crop_frame(cax, color=WINDOW_COLOR[tag])

    # ==== spectra, one plot per row, shared axes; y in dB (10 log10 of the a.u. power), x in plain decimals
    axs, top_curves = [], {}
    for ri, col in enumerate(cols):
        top = IMG_TOP + ri * (F + rowgap)
        ax = add_ax(fig, spec_x, top, spec_w, F)
        if axs:
            ax.sharex(axs[0])
            ax.sharey(axs[0])
        axs.append(ax)
        cs.clean_axes(ax, grid="y")
        peak = None
        for ck in ("krea", "dev"):
            f, p = whitened_spectrum(col[ck])
            ok = f > 2.5e-3
            ax.plot(f[ok], p[ok], color=CK_INK[ck], lw=1.0, solid_capstyle="round", zorder=3 if ck == "dev" else 2)
            if ri == 0:
                top_curves[ck] = (f[ok], p[ok])
            if ck == "dev":
                band = (f > 1 / 128) & (f < 1 / 4)
                i = np.argmax(np.where(band, p, -np.inf))
                peak = (f[i], p[i])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([peak[0]], [peak[1]], marker="o", ms=2.6, color=cs.INK, lw=0, zorder=4)
        ax.annotate(f"{1 / peak[0]:.0f} px", xy=peak, xytext=(0, 3.0), textcoords="offset points",
                    fontsize=cs.FS_VALUE, color=cs.INK, ha="center", va="bottom")
        # peak ratios belong to the caption; this line records which arm owns each number
        if col["key"] == "official":
            r = peak_ratio(col["dev"], col["krea"])
            txt = f"peak ratio {r:.0f}× (this pair only)"
        else:
            r = np.log([peak_ratio(a, b) for a, b in zip(col["dev_all"], col["krea_all"])])
            gm, lo, hi = np.exp(cs.boot(r))
            txt = f"8 prompts: peak ratio {gm:.0f}× [{lo:.0f}, {hi:.0f}]"
            print(col["key"], "per-prompt peak ratios", np.round(np.exp(r), 1))
        print(col["key"], txt, f"period {1 / peak[0]:.1f} px")
        if ri < len(cols) - 1:
            ax.tick_params(labelbottom=False)
    axs[0].set_xlim(2.5e-3, 0.5)
    axs[0].set_ylim(5e-8, 6e-3)
    axs[0].xaxis.set_major_locator(FixedLocator([0.01, 0.1]))
    axs[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].xaxis.set_minor_locator(NullLocator())
    axs[0].yaxis.set_major_locator(FixedLocator([1e-7, 1e-5, 1e-3]))
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{10 * np.log10(v):.0f}".replace("-", "−")))
    axs[0].yaxis.set_minor_locator(NullLocator())
    axs[1].set_ylabel("Power (dB)", fontsize=cs.FS_LABEL, labelpad=3)
    axs[-1].set_xlabel("Frequency (cycles/px)", fontsize=cs.FS_LABEL, labelpad=2)

    # direct labels on the top spectra: gray = Krea above its flat left end; black = FLUX.1-dev in the free upper
    # right, beside the falling shoulder of its peak (the black curve is below -38 dB right of f = 0.08)
    f, p = top_curves["krea"]
    x_lab = 3.1e-3
    axs[0].text(x_lab, np.interp(x_lab, f, p) * 10 ** 0.45, "Krea", fontsize=cs.FS_LABEL, color=cs.INK,
                ha="left", va="bottom")
    f, p = top_curves["dev"]
    x_dev = 0.085
    assert np.interp(x_dev, f, p) < 10 ** -3.8, "FLUX.1-dev label would touch its curve"
    axs[0].text(x_dev, 10 ** -3.3, "FLUX.1-dev", fontsize=cs.FS_LABEL, color=cs.INK, ha="left", va="bottom",
                zorder=2.5, bbox=dict(fc="white", ec="none", pad=0.6))   # breaks the -30 dB gridline behind it

    # column labels under the bottom image row, on the line of the spectra tick labels
    fig.canvas.draw()
    tick = axs[-1].xaxis.get_ticklabels()[0].get_window_extent(fig.canvas.get_renderer())
    for ck, name in (("dev", "FLUX.1-dev"), ("krea", "FLUX.1-Krea-dev")):
        fig.text((x_ck[ck] + cell / 2) / W, tick.y0 / fig.bbox.height, name, fontsize=cs.FS_LABEL, color=cs.INK,
                 ha="center", va="bottom")

    cs.panel_tag(fig, 0.0, 1 - TAG_Y / H, "a", "Samples")
    cs.panel_tag(fig, x_b / W, 1 - TAG_Y / H, "b", "Whitened spectra")
    print(f"figure size {W:.2f} x {H:.3f} in; frames {F:.3f} in, crops {C:.3f} in, spectra {spec_w:.2f} x {F:.3f} in")
    cs.finish(fig, str(cs.HERE / "f5_checkpoint"), photo=True)
    print("prompt:", prompt)


if __name__ == "__main__":
    main()
