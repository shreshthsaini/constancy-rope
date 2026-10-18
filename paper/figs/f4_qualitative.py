"""F4: qualitative 4K comparison, held-out manifest rows 8, 9, 10 (the first three held-out rows,
fixed before any output was viewed), seed 0, six methods.

Per prompt: Lanczos-reduced full frames (small) and two native-pixel crops at the same location for
every method. A = maximum Laplacian energy of the across-method mean luminance over all six methods
(qual_helpers.detail_box, tiles overlapping B excluded); B = image center. Crops are 384 px and are
never resampled or sharpened.

Regenerate: OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f4_qualitative.py
"""
from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import crope_style as cs
import qual_helpers as qh

ROWS = (8, 9, 10)
SHORT = {8: "A stone bridge with arched spans", 9: "A panoramic view of Paris",
         10: "A cluttered artist's corner"}
GROUPS = [["ntk2", "dype"], ["sega_official", "temp_only_sbase"], ["constancy_tm_sbase", "sega_ctm_v2"]]
METHODS = [m for grp in GROUPS for m in grp]
CROP = 384
THUMB = 384
W = cs.TEXT
HALO = [pe.withStroke(linewidth=1.6, foreground="#111317")]

# layout (inches)
LC = 0.27            # rotated prompt labels
GI, GG = 2 / 72, 0.09  # gutter inside a method group, gap between groups
C = (W - LC - 3 * GI - 2 * GG) / 6
F = 0.60             # full-frame thumbnail edge
HEAD = 0.20          # method-name header
VG = 0.045           # full frames to crops
BG = 0.09            # between prompt blocks
FOOT = 0.16          # note line under the grid
BLOCK = F + VG + 2 * C + GI
H = HEAD + 3 * BLOCK + 2 * BG + FOOT


def col_x(j):
    g, k = divmod(j, 2)
    return LC + g * (2 * C + GI + GG) + k * (C + GI)


def axes_in(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def photo(ax, img, frame=False):
    ax.imshow(img, interpolation="none")
    ax.axis("off")
    if frame:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=cs.RULE,
                               linewidth=0.4, zorder=5, clip_on=False))


def crop_box(ax, box, scale, letter=None):
    x0, y0, x1, y1 = (v * scale for v in box)
    r = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=0.6)
    r.set_path_effects([pe.withStroke(linewidth=1.8, foreground="#111317"), pe.Normal()])
    ax.add_patch(r)
    if letter:
        right = x1 + 0.03 * THUMB < THUMB * 0.9
        ax.text(x1 + 0.025 * THUMB if right else x0 - 0.025 * THUMB, y0, letter, color="white",
                fontsize=5.5, fontweight="bold", ha="left" if right else "right", va="top",
                path_effects=HALO)


def main():
    cs.style()
    table = cs.heldout()
    prompts = cs.prompts()
    fig = plt.figure(figsize=(W, H))
    for j, m in enumerate(METHODS):
        fig.text((col_x(j) + C / 2) / W, (H - 0.12) / H, cs.label(m), color=cs.text_color(m), fontsize=6.5,
                 fontweight="bold", ha="center", va="baseline")
        fig.add_artist(plt.Line2D([col_x(j) / W, (col_x(j) + C) / W], [(H - 0.152) / H] * 2,
                                  color=cs.color(m), lw=1.1, solid_capstyle="butt"))
    for b, row in enumerate(ROWS):
        src = qh.sources(row, METHODS, table=table)
        mean_y = qh.mean_luma([src[m] for m in METHODS])
        box_b = qh.center_box(CROP)
        box_a = qh.detail_box(mean_y, CROP, exclude=[box_b])
        print(f"row {row}: A {box_a}  B {box_b}")
        top = H - HEAD - b * (BLOCK + BG)
        fig.add_artist(Rectangle((LC / W, (top - F - 0.03) / H), (W - LC) / W, (F + 0.06) / H,
                                 transform=fig.transFigure, facecolor=cs.PANEL, edgecolor="none", zorder=-1))
        for j, m in enumerate(METHODS):
            x = col_x(j)
            ax = axes_in(fig, x + (C - F) / 2, top - F, F, F)
            photo(ax, cs.thumb(src[m], THUMB))
            s = THUMB / 4096
            crop_box(ax, box_a, s, "A" if j == 0 else None)
            crop_box(ax, box_b, s, "B" if j == 0 else None)
            for i, (box, letter) in enumerate(((box_a, "A"), (box_b, "B"))):
                y = top - F - VG - (i + 1) * C - i * GI
                axc = axes_in(fig, x, y, C, C)
                photo(axc, cs.crop(src[m], box), frame=True)
                if j == 0:
                    axc.text(0.05, 0.95, letter, transform=axc.transAxes, color="white", fontsize=7,
                             fontweight="bold", ha="left", va="top", path_effects=HALO)
        assert prompts[row]["prompt"].startswith(SHORT[row])
        yc = top - BLOCK / 2
        fig.text(0.07 / W, yc / H, f"Row {row}", rotation=90, ha="center", va="center", fontsize=7,
                 fontweight="bold", color=cs.INK)
        fig.text(0.19 / W, yc / H, SHORT[row] + "…", rotation=90, ha="center", va="center",
                 fontsize=6.5, color=cs.INK2)
    fig.text(W / W, 0.035 / H, f"A: highest-detail tile of the across-method mean luminance;  B: image center;  "
             f"native {CROP} px crops, never resampled", ha="right", va="baseline", fontsize=6.5, color=cs.INK2)
    cs.finish(fig, cs.HERE / "f4_qualitative", photo=True)


if __name__ == "__main__":
    main()
