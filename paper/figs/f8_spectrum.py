"""F8: luminance power spectrum error by octave versus Aesthetic-4K references, and the paired difference from SEGA.

Definition (crope/metrics/image.py, octave_profile and profile_error; evaluate.py diagnostics): linear-sRGB luminance
(Rec. 709 weights), mean removed, unwindowed 2-D FFT, power |F|^2/(H W), radially averaged within octave bands with
edges 2^-12 ... 2^-1 cycles per pixel (band b_i covers wavelengths 2^(11-i) to 2^(12-i) px), natural log. The error is
signed: ln P_method(b) minus the mean ln P(b) over the 195 Aesthetic-4K 4096-px reference images.

(a) mean error per method with 95% percentile bootstrap intervals over prompts (2,000 resamples, crope_style.boot);
(b) paired difference from released SEGA on the same prompt and seed (crope_style.paired). Seed 0, rows 8-194, n = 187.

Regenerate (from paper/figs):  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f8_spectrum.py
"""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import crope_style as cs

A = ["ntk2", "dype", "sega_official", "temp_only_sbase", "constancy_tm_sbase", "sega_ctm_v2"]
B = ["ntk2", "dype", "temp_only_sbase", "constancy_tm_sbase", "sega_ctm_v2"]
BANDS = [f"spectrum_log_error_b{i:02d}" for i in range(11)]
METRICS_JSON = cs.SCR / "eval_krea" / "main2" / "metrics.json"

# geometry (inches)
W = cs.TEXT
LEFT = 0.62          # tick labels + two-line y label
RIGHT = 0.05
TOP = 0.04
KEYH = 0.30          # two-row key above panel (a)
LETTER = 0.15        # panel-letter line above each panel
HA = 1.15
GAP = 0.20
HB = 1.15
BOTTOM = 0.40
H = TOP + KEYH + LETTER + HA + GAP + HB + BOTTOM
PWIDTH = W - LEFT - RIGHT


def band_axis():
    """Band centres from the evaluation record, as log2(cycles per pixel)."""
    st = json.load(open(METRICS_JSON))["status"]["spectrum"]
    f = np.asarray(st["frequency"], float)
    assert len(f) == len(BANDS) and np.allclose(f, 2.0 ** (np.arange(-12, -1) + 0.5))
    return np.log2(f), int(st["reference_count"])


def compute():
    h = cs.heldout()
    assert h.idx.between(8, 194).all() and (h.seed == 0).all()
    mean = {}
    for m in A:
        d = h[h.method == m]
        assert len(d) == 187 and d[BANDS].notna().all().all()
        mean[m] = np.array([cs.boot(d[b]) for b in BANDS])           # (11, 3): mean, lo, hi
    diff = {}
    for m in B:
        rows = [cs.paired(h, m, b) for b in BANDS]
        assert all(r[3] == 187 for r in rows)
        diff[m] = np.array([r[:3] for r in rows])
    return mean, diff


def key(fig, x0, y0, entries, ncol=3, pitch=0.14, colgap=0.26, size=6.8):
    """Figure key in the fixed method order, row-major in `ncol` columns: a short line with the method's
    marker, then the name in ink. x0 is the left edge and y0 the first row's centre, inches from the top."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def width(text):
        t = fig.text(0, 0, text, fontsize=size)
        w = t.get_window_extent(rend).width / fig.dpi
        t.remove()
        return w

    cols = [entries[c::ncol] for c in range(ncol)]
    colw = [max(width(cs.label(m)) for m in col) for col in cols]
    x = x0
    for c, col in enumerate(cols):
        for r, m in enumerate(col):
            spec = cs.METHODS[m]
            y = 1 - (y0 + r * pitch) / H
            fig.add_artist(Line2D([x / W, (x + 0.17) / W], [y] * 2, color=spec["color"],
                                  lw=1.6 if m == "sega_official" else 1.1, solid_capstyle="butt"))
            fig.add_artist(Line2D([(x + 0.085) / W], [y], ls="none", marker=spec["marker"],
                                  ms=2.9 if spec["marker"] == "o" else 2.6, mfc=spec["color"], mec="white",
                                  mew=0.35))
            fig.text((x + 0.23) / W, y, cs.label(m), fontsize=size, color=cs.INK, ha="left", va="center")
        x += 0.23 + colw[c] + colgap


def series(ax, x, arr, key_, emphasis=False, band_alpha=0.13):
    spec = cs.METHODS[key_]
    c = spec["color"]
    ax.fill_between(x, arr[:, 1], arr[:, 2], color=c, alpha=band_alpha, lw=0, zorder=2)
    ax.plot(x, arr[:, 0], color=c, lw=1.6 if emphasis else 1.1, zorder=4 if emphasis else 3,
            solid_joinstyle="round")
    ax.plot(x, arr[:, 0], ls="none", marker=spec["marker"], ms=2.9 if spec["marker"] == "o" else 2.6,
            mfc=c, mec="white", mew=0.35, zorder=5 if emphasis else 4)


def style_axis(ax, ylim, yticks):
    ax.set_xlim(-12, -1)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _p: "0" if abs(v) < 1e-9 else f"{v:.1f}".replace("-", "−")))
    ax.set_xticks(np.arange(-12, 0, 2))
    ax.set_xticks(np.arange(-11, 0, 2), minor=True)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{2 ** -v:.0f}"))
    ax.grid(axis="y", color=cs.GRIDC, lw=0.4)
    ax.grid(axis="x", which="major", color=cs.GRIDC, lw=0.4)
    ax.tick_params(axis="both", which="major", length=2.2, width=0.6, pad=1.8, labelsize=7, color=cs.INK2)
    ax.tick_params(axis="x", which="minor", length=1.3, width=0.5, color=cs.INK2)
    ax.axhline(0, color=cs.RULE, lw=0.9, zorder=1)


def main():
    cs.style()
    x, n_ref = band_axis()
    mean, diff = compute()
    fig = plt.figure(figsize=(W, H))
    ax_a = fig.add_axes([LEFT / W, 1 - (TOP + KEYH + LETTER + HA) / H, PWIDTH / W, HA / H])
    ax_b = fig.add_axes([LEFT / W, BOTTOM / H, PWIDTH / W, HB / H], sharex=ax_a)

    # (a) error versus the reference
    for m in A:
        series(ax_a, x, mean[m], m, emphasis=(m == "sega_official"), band_alpha=0.11)
    style_axis(ax_a, (-1.6, 1.6), np.arange(-1.5, 1.6, 0.5))
    ax_a.tick_params(axis="x", labelbottom=False)
    ax_a.set_ylabel("Error vs. reference\nln $P$ − ln $P_\\mathrm{ref}$", fontsize=7.5,
                    linespacing=1.3)
    # (b) paired difference from SEGA
    for m in B:
        series(ax_b, x, diff[m], m, band_alpha=0.18)
    style_axis(ax_b, (-1.85, 0.55), np.arange(-1.5, 0.6, 0.5))
    ax_b.set_ylabel("Difference from SEGA\nln $P$ − ln $P_\\mathrm{SEGA}$", fontsize=7.5, linespacing=1.3)
    ax_b.set_xlabel("Wavelength (px), one point per octave band", fontsize=7.5)
    for ax in (ax_a, ax_b):
        ax.yaxis.set_label_coords(-0.30 / PWIDTH, 0.5)

    # the two octaves where removing SEGA's per-band allocation (SEGA base + temp. minus SEGA) adds the most power
    top2 = np.sort(np.argsort(diff["temp_only_sbase"][:, 0])[-2:])
    assert top2[1] - top2[0] == 1, "highlight expects two adjacent octaves"
    e0, e1 = x[top2[0]] - 0.5, x[top2[1]] + 0.5                       # band edges, log2 cycles per pixel
    for ax in (ax_a, ax_b):
        ax.axvspan(e0, e1, color=cs.PANEL, lw=0, zorder=0)
    ax_a.text((e0 + e1) / 2, 1.43, f"{2 ** -e1:.0f} to {2 ** -e0:.0f} px", ha="center", va="center",
              fontsize=6.5, color=cs.INK2)
    print("highlight bands", top2.tolist(), "temp - SEGA", np.round(diff["temp_only_sbase"][top2], 3).tolist())
    key(fig, LEFT, TOP + 0.08, A)
    # reading cues for the sign, in the empty left corners of each panel
    for ax, what in ((ax_a, f"Aesthetic-4K ({n_ref} images)"), (ax_b, "SEGA (released)")):
        ax.text(0.012, 0.965, f"\u2191 more power than {what}", transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, color=cs.INK2)
        ax.text(0.012, 0.035, "\u2193 less power", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.5, color=cs.INK2)
    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        cs.panel_tag(ax, -0.60 / PWIDTH, 1.0 + 0.07 / ax.bbox.height * fig.dpi, letter)
    nyquist = np.log2(0.5 * 224 / 4096)
    for ax in (ax_a, ax_b):
        ax.axvline(nyquist, color=cs.INK2, ls=":", lw=.8)
    ax_b.text(nyquist + .12, -1.65, "224-view Nyquist", rotation=90,
              fontsize=6.3, color=cs.INK2, va="bottom")
    cs.finish(fig, cs.HERE / "f8_spectrum")
    main_panel(x, diff)



def main_panel(x, diff):
    fig, ax = plt.subplots(figsize=(cs.TEXT, 1.92))
    fig.subplots_adjust(left=.095, right=.99, bottom=.23, top=.77)
    for m in B:
        series(ax, x, diff[m], m, emphasis=m == "temp_only_sbase")
    style_axis(ax, (-1.85, .55), [-1.5, -1, -.5, 0, .5])
    ax.set_xlabel("Wavelength (px)", fontsize=7.5, labelpad=2)
    ax.set_ylabel(r"Paired $\Delta\ln P$ vs. SEGA", fontsize=7.5)
    ax.axvspan(-5, -3, color=cs.PANEL, lw=0, zorder=0)
    nyquist = np.log2(.5 * 224 / 4096)
    ax.axvline(nyquist, color=cs.INK2, ls=":", lw=.8)
    ax.text(nyquist - .12, -1.65, "224-view Nyquist", ha="right", fontsize=6.5, color=cs.INK2)
    ax.text(-4, .43, "8 to 32 px", ha="center", fontsize=6.5, color=cs.INK2)
    handles = [Line2D([], [], color=cs.color(m), marker=cs.METHODS[m]["marker"],
                       lw=1.2, ms=3, label=cs.label(m)) for m in B]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=7, handlelength=1.5, columnspacing=1.4)
    cs.finish(fig, cs.HERE / "f8_spectrum_main")


if __name__ == "__main__":
    main()
