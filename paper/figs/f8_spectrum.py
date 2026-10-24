"""F8: luminance power spectrum error by octave versus Aesthetic-4K references, and the paired difference from SEGA.

Definition (crope/metrics/image.py, octave_profile and profile_error; evaluate.py diagnostics): linear-sRGB luminance
(Rec. 709 weights), mean removed, unwindowed 2-D FFT, power |F|^2/(H W), radially averaged within octave bands with
edges 2^-12 ... 2^-1 cycles per pixel (band b_i covers wavelengths 2^(11-i) to 2^(12-i) px), natural log. The error is
signed: ln P_method(b) minus the mean ln P(b) over the 195 Aesthetic-4K 4096-px reference images.

Two outputs from the same numbers (seed 0, held-out rows 8-194, n = 187):
- f8_spectrum.pdf (appendix A.3, 5.50 in wide, included at \\linewidth): (a) mean error per method with 95% percentile
  bootstrap intervals over prompts (2,000 resamples, crope_style.boot); (b) paired difference from released SEGA on
  the same prompt and seed (crope_style.paired).
- f8_spectrum_main.pdf (main text F4, 2.75 x 2.00 in, a wrapfigure included at 0.5\\linewidth): panel (b) alone.
The gray band marks the two octaves where removing SEGA's per-band allocation (SEGA base + temp. minus SEGA) adds the
most power (8 to 32 px, asserted); the dashed vertical marks the Nyquist frequency of a 224-pixel view; a dashed blue
zero line is SEGA (released). Text inside: names, symbols and axis numbers only; the caption carries the rest.

Regenerate (from paper/figs):  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f8_spectrum.py
"""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import crope_style as cs
import orx_figstyle

A = ["ntk2", "dype", "sega_official", "temp_only_sbase", "constancy_tm_sbase", "sega_ctm_v2"]
B = ["ntk2", "dype", "temp_only_sbase", "constancy_tm_sbase", "sega_ctm_v2"]
BANDS = [f"spectrum_log_error_b{i:02d}" for i in range(11)]
METRICS_JSON = cs.SCR / "eval_krea" / "main2" / "metrics.json"
NYQUIST = float(np.log2(0.5 * 224 / 4096))      # 224-pixel view, in log2 cycles per original pixel

BAND_ALPHA = 0.18                                # CI bands (R4: an 18% band for curves)
BAND_ALPHA_A = 0.13                              # A.3 (a): six bands overlap near 0, so a lighter tint keeps them apart
LW, LW_EMPH = 1.0, 1.4
MSIZE = {"o": 3.4, "s": 3.0, "D": 2.9, "^": 3.9, "v": 3.9}   # equal visual weight across shapes (points)
DASH = (0, (3.0, 1.8))                           # reference lines: SEGA zero line and the Nyquist marker
LABEL_PT, TICK_PT, NOTE_PT = 8.0, 7.0, 7.0
YLAB_IN = 0.29          # right edge of a rotated y label, inches left of the axes (clear of the tick labels)
BAND_TEXT = "8–32 px"


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


def highlight(x, diff):
    """Band edges (log2 cycles per pixel) of the two adjacent octaves where SEGA base + temp. minus SEGA is largest."""
    top2 = np.sort(np.argsort(diff["temp_only_sbase"][:, 0])[-2:])
    assert top2[1] - top2[0] == 1, "highlight expects two adjacent octaves"
    e0, e1 = x[top2[0]] - 0.5, x[top2[1]] + 0.5
    assert (2 ** -e1, 2 ** -e0) == (8, 32), "the band label and the captions say 8 to 32 px"
    print("highlight bands", top2.tolist(), "temp - SEGA", np.round(diff["temp_only_sbase"][top2, 0], 3).tolist())
    return e0, e1


# ------------------------------------------------------------------ drawing helpers
def line_kw(key, emphasis=False):
    spec = cs.METHODS[key]
    c = spec["color"]
    return dict(color=c, ls=spec["ls"], lw=LW_EMPH if emphasis else LW, marker=spec["marker"],
                ms=MSIZE[spec["marker"]], mfc=c, mec="white", mew=0.4, solid_joinstyle="round",
                dash_joinstyle="round")


def series(ax, x, arr, key, emphasis=False, alpha=BAND_ALPHA):
    c = cs.METHODS[key]["color"]
    ax.fill_between(x, arr[:, 1], arr[:, 2], color=c, alpha=alpha, lw=0, zorder=2)
    ax.plot(x, arr[:, 0], zorder=4 if emphasis else 3, **line_kw(key, emphasis))


def style_axis(ax, ylim, yticks):
    ax.set_xlim(-12, -1)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _p: "0" if abs(v) < 1e-9 else f"{v:.1f}".replace("-", "−")))
    ax.set_xticks(np.arange(-12, 0, 2))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{2 ** -v:.0f}"))
    cs.clean_axes(ax, grid="y")
    ax.grid(True, axis="y", color=cs.GRIDC, lw=0.4)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.6, pad=1.8, labelsize=TICK_PT)


def region_marks(ax, e0, e1):
    """Gray band for 8 to 32 px and the dashed 224-view Nyquist vertical (no text)."""
    ax.axvspan(e0, e1, color=cs.PANEL, lw=0, zorder=0)
    ax.axvline(NYQUIST, color=cs.INK2, lw=0.6, ls=DASH, zorder=1)


def sega_zero(ax):
    """The zero line of a paired difference is the reference arm, SEGA (released): dashed, method hue."""
    ax.axhline(0, color=cs.METHOD, lw=0.6, ls=DASH, zorder=2.5)


def legend_row(fig, keys, y_in, emph=(), handle_in=0.22, pad_in=0.045, gap_in=0.10):
    """One unframed legend row centred on the figure, its centre `y_in` inches below the top edge."""
    em = LABEL_PT / 72
    handles = [Line2D([], [], label=cs.label(k), **line_kw(k, k in emph)) for k in keys]
    leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 1 - y_in / fig.get_figheight()),
                     ncol=len(keys), frameon=False, fontsize=LABEL_PT, borderpad=0, borderaxespad=0,
                     handlelength=handle_in / em, handletextpad=pad_in / em, columnspacing=gap_in / em)
    return leg   # Figure.legend keeps every legend it makes; adding it again would draw it twice


def check_inside(fig, artists, margin_in=0.03):
    """Every legend and label stays inside the canvas with a margin (the audit only checks text)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for a in artists:
        bb = a.get_window_extent(r)
        x0, x1 = bb.x0 / fig.dpi, bb.x1 / fig.dpi
        print(f"  legend/label spans {x0:.3f} to {x1:.3f} in of {fig.get_figwidth():.2f} in")
        assert x0 >= margin_in and x1 <= fig.get_figwidth() - margin_in, (a, x0, x1)


def clearance(ax, text, x, curves):
    """Smallest vertical gap (inches) between a text box and any curve (line, CI band edge or marker) under it."""
    fig = ax.figure
    fig.canvas.draw()
    bb = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    xs = np.linspace(bb.x0, bb.x1, 200)
    gaps = []
    for arr in curves:
        for col in (0, 1, 2):
            ys = np.interp(xs, x, arr[:, col])
            gaps.append(np.min(np.where(ys < bb.y0, bb.y0 - ys, np.where(ys > bb.y1, ys - bb.y1, -1.0))))
        inside = (x >= bb.x0) & (x <= bb.x1)
        for y in arr[inside, 0]:
            gaps.append(bb.y0 - y if y < bb.y0 else (y - bb.y1 if y > bb.y1 else -1.0))
    in_per_unit = ax.bbox.height / fig.dpi / (ax.get_ylim()[1] - ax.get_ylim()[0])
    return min(gaps) * in_per_unit


def finish_at_width(fig, stem):
    """cs.finish at a wrapfigure width. orx_figstyle's audit knows the 3.25 / 5.50 / 6.75 in widths; 2.75 in
    (0.5\\linewidth) is this figure's final print width, so it is registered for this one save and every other
    audit check still runs."""
    saved = orx_figstyle.COLUMN
    orx_figstyle.COLUMN = fig.get_figwidth()
    try:
        cs.finish(fig, stem)
    finally:
        orx_figstyle.COLUMN = saved


# ------------------------------------------------------------------ A.3: two panels, full width
def appendix_figure(x, mean, diff, e0, e1):
    W, H = cs.TEXT, 2.45
    LEFT, MID, RIGHT = 0.43, 0.57, 0.06          # y tick labels and math y label left of each panel
    TOP_AX, BOTTOM = 0.43, 0.33                  # legend row and panel tags above; ticks and x label below
    PW = (W - LEFT - MID - RIGHT) / 2
    PH = H - TOP_AX - BOTTOM
    fig = plt.figure(figsize=(W, H))
    ax_a = fig.add_axes([LEFT / W, BOTTOM / H, PW / W, PH / H])
    xb = LEFT + PW + MID
    ax_b = fig.add_axes([xb / W, BOTTOM / H, PW / W, PH / H], sharex=ax_a)

    # (a) error versus the Aesthetic-4K reference mean (zero line: the reference, a light rule)
    for ax in (ax_a, ax_b):
        region_marks(ax, e0, e1)
    ax_a.axhline(0, color=cs.RULE, lw=0.7, zorder=1)
    for m in A:
        series(ax_a, x, mean[m], m, emphasis=(m == "sega_official"), alpha=BAND_ALPHA_A)
    style_axis(ax_a, (-1.6, 1.6), np.arange(-1.5, 1.6, 0.5))
    ax_a.set_ylabel(r"$\ln P - \ln P_{\mathrm{ref}}$", fontsize=LABEL_PT, labelpad=3)
    # (b) paired difference from SEGA (released), the dashed blue zero line
    sega_zero(ax_b)
    for m in B:
        series(ax_b, x, diff[m], m, emphasis=(m == "temp_only_sbase"))
    style_axis(ax_b, (-1.85, 0.55), np.arange(-1.5, 0.6, 0.5))
    ax_b.set_ylabel(r"$\ln P - \ln P_{\mathrm{SEGA}}$", fontsize=LABEL_PT, labelpad=3)
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Wavelength (px)", fontsize=LABEL_PT, labelpad=2)
        ax.yaxis.set_label_coords(-YLAB_IN / PW, 0.5)

    band = ax_a.text((e0 + e1) / 2, 1.55, BAND_TEXT, ha="center", va="top", fontsize=NOTE_PT, color=cs.INK2,
                     zorder=6)
    gap = clearance(ax_a, band, x, [mean[m] for m in A])
    print(f"A.3 band label clearance {gap:.3f} in")
    assert gap > 0.03

    # one legend row above both panels, panel tags on a shared baseline above each panel
    leg = legend_row(fig, A, 0.10, emph=("sega_official",), handle_in=0.16, pad_in=0.04, gap_in=0.095)
    tag_y = 1 - (TOP_AX - 0.08) / H
    cs.panel_tag(fig, 0.0, tag_y, "a", "Error vs. reference")
    cs.panel_tag(fig, (xb - LEFT) / W, tag_y, "b", "Paired difference")
    check_inside(fig, [leg], margin_in=0.02)
    cs.finish(fig, cs.HERE / "f8_spectrum")


# ------------------------------------------------------------------ F4: panel (b) alone, wrapfigure
def main_figure(x, diff, e0, e1):
    W, H = cs.TEXT / 2, 2.00
    LEFT, RIGHT, TOP_AX, BOTTOM = 0.42, 0.06, 0.38, 0.33
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([LEFT / W, BOTTOM / H, (W - LEFT - RIGHT) / W, (H - TOP_AX - BOTTOM) / H])
    region_marks(ax, e0, e1)
    sega_zero(ax)
    for m in B:
        series(ax, x, diff[m], m, emphasis=(m == "temp_only_sbase"))
    style_axis(ax, (-1.85, 0.75), [-1.5, -1.0, -0.5, 0, 0.5])
    ax.set_xlabel("Wavelength (px)", fontsize=LABEL_PT, labelpad=2)
    ax.set_ylabel(r"$\ln P - \ln P_{\mathrm{SEGA}}$", fontsize=LABEL_PT)   # names the reference: no in-plot label
    ax.yaxis.set_label_coords(-YLAB_IN / ax.bbox.width * fig.dpi, 0.5)

    band = ax.text((e0 + e1) / 2, 0.72, BAND_TEXT, ha="center", va="top", fontsize=NOTE_PT, color=cs.INK2, zorder=6)
    gap = clearance(ax, band, x, [diff[m] for m in B])
    print(f"F4 band label clearance {gap:.3f} in")
    assert gap > 0.03

    rows = (["ntk2", "dype", "temp_only_sbase"], ["constancy_tm_sbase", "sega_ctm_v2"])
    legs = [legend_row(fig, keys, 0.09 + i * 0.15, emph=("temp_only_sbase",), handle_in=0.25)
            for i, keys in enumerate(rows)]
    check_inside(fig, legs, margin_in=0.02)
    finish_at_width(fig, cs.HERE / "f8_spectrum_main")


def main():
    cs.style()
    cs.script_floor(LABEL_PT)      # the ref and SEGA subscripts of the 8 pt y labels print at 6.5 pt, not 5.6
    x, n_ref = band_axis()
    assert n_ref == 195
    mean, diff = compute()
    e0, e1 = highlight(x, diff)
    appendix_figure(x, mean, diff, e0, e1)
    main_figure(x, diff, e0, e1)


if __name__ == "__main__":
    main()
