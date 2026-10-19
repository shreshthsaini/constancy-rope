"""F4: qualitative 4K comparisons from the original first held-out rows, seed 0.

Default main figure: rows 8 and 9, six methods, small full frames and one common
384px crop per row. Crop selection maximizes minimum normalized per-arm detail;
its exact rule, sources and boxes are saved beside the figure. This crop selection
was introduced during review after historical outputs were available.

The --appendix option retains the original three-row, two-crop layout and mean-image
Laplacian rule. It regenerates f4_qualitative.pdf only when explicitly requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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


def appendix():
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


def main():
    cs.style()
    table, prompts = cs.heldout(), cs.prompts()
    main_rows = (8, 9)
    header, full, vg, gap, foot = .22, .50, .04, .12, .18
    block = full + vg + C
    height = header + 2 * block + gap + foot
    fig = plt.figure(figsize=(W, height))
    def ax_at(x, y, width, h):
        return fig.add_axes([x / W, y / height, width / W, h / height])
    for j, method in enumerate(METHODS):
        fig.text((col_x(j) + C / 2) / W, (height - .12) / height, cs.label(method),
                 fontsize=6.5, fontweight='bold', ha='center', va='baseline', color=cs.text_color(method))
        fig.add_artist(plt.Line2D([col_x(j)/W, (col_x(j)+C)/W], [(height-.16)/height]*2,
                                 color=cs.color(method), lw=1.1, solid_capstyle='butt'))
    receipts = []
    for block_index, row in enumerate(main_rows):
        src = qh.sources(row, METHODS, table=table)
        box, audit = qh.common_detail_box([src[m] for m in METHODS], CROP)
        audit.update(row=row, prompt_id=prompts[row]['id'], prompt=prompts[row]['prompt'],
                     methods=METHODS, seed=0,
                     source_sha256={m: hashlib.sha256(src[m].read_bytes()).hexdigest() for m in METHODS})
        receipts.append(audit)
        print(f"main row {row}: A {box}; minimum normalized energy {audit['minimum_normalized_energy']:.6f}")
        top = height - header - block_index*(block+gap)
        fig.add_artist(Rectangle((LC/W, (top-full-.025)/height), (W-LC)/W, (full+.05)/height,
                                 transform=fig.transFigure, facecolor=cs.PANEL, edgecolor='none', zorder=-1))
        for j, method in enumerate(METHODS):
            x = col_x(j)
            ax = ax_at(x+(C-full)/2, top-full, full, full)
            photo(ax, cs.thumb(src[method], THUMB))
            crop_box(ax, box, THUMB/4096, 'A' if j == 0 else None)
            cropped = ax_at(x, top-full-vg-C, C, C)
            photo(cropped, cs.crop(src[method], box), frame=True)
            if j == 0:
                cropped.text(.05,.95,'A',transform=cropped.transAxes,color='white',fontsize=7,
                             fontweight='bold',ha='left',va='top',path_effects=HALO)
        assert prompts[row]['prompt'].startswith(SHORT[row])
        center = top-block/2
        fig.text(.07/W, center/height, f'Row {row}', rotation=90, ha='center',va='center',
                 fontsize=7,fontweight='bold',color=cs.INK)
        fig.text(.19/W,center/height,SHORT[row]+'…',rotation=90,ha='center',va='center',
                 fontsize=6.5,color=cs.INK2)
    fig.text(1,.045/height,'A: highest minimum normalized detail across all arms; native 384 px, identical coordinates',
             ha='right',va='baseline',fontsize=6.3,color=cs.INK2)
    # Keep native crop pixels lossless. Full-frame thumbnail resize is the only resampling.
    cs.finish(fig,cs.HERE/'f4_qualitative_main',formats=('pdf','svg'))
    legacy = cs.HERE/'f4_qualitative.pdf'
    manifest = {
        'schema': 'qualitative-common-detail-v1', 'rows': receipts,
        'method_order': METHODS, 'printed_size_inches': [W,height],
        'crop_pixels': 'unaltered RGB native pixels, lossless embedded images',
        'full_frame_thumbnails': '384x384 Lanczos reduction of original full frame',
        'legacy_appendix': {'path': str(legacy), 'sha256': hashlib.sha256(legacy.read_bytes()).hexdigest(),
            'original_generator_sha256': '87668e4884a955500ca3b604b5a1922a77b932d39b9dcbaa679065bda3e48a20',
            'original_helper_sha256': '685413fa49ef123cacc08a964a15df2d3403ed95169de34aaef4d33eb2a9efa7',
            'source_git_commit': '4bf534d16b821f45355b6e5354cfd1c6bdbe0448',
            'regenerate': 'python paper/figs/f4_qualitative.py --appendix'},
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(Path(qh.__file__).read_bytes()).hexdigest(),
    }
    (cs.HERE/'f4_qualitative_main_crops.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--appendix',action='store_true',help='Regenerate historical full appendix figure instead')
    args = parser.parse_args()
    (appendix if args.appendix else main)()
