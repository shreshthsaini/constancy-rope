"""Figure 5 (main text, Section 5.3): two-row qualitative grid at 4096 px, seed 0.

Selection rule, fixed before any output was viewed: the first two held-out manifest rows (8 and 9;
the held-out split starts at row 8). Figure A.2 (f4_qualitative_main.pdf) shows the same two rows
plus row 10 with the same crops.

Layout: prompts are rows, arms are columns (NTK, DyPE | SEGA (released), SEGA base + temp. |
Constancy-TM, SEGA + TM; pairs follow the palette groups). Each cell is the Lanczos-reduced full
frame with a 1 pt red box, and directly below it the unaltered native 384-px crop at that box with
a 1 pt red frame (gold standard, Section 6). Prompt tags sit at the left in 8 pt sans (rotated),
arm names once at the bottom.

Crop location per row: the review-stage common-detail rule of Figure A.2
(qual_helpers.common_detail_box: maximize, over a 96-px-stride grid with a 64-px margin, the least
across-arm mean-squared Laplacian, each arm divided by its 90th percentile over candidates). The
script recomputes it and asserts that the boxes equal those recorded in
f4_qualitative_main_crops.json, so the body grid and Figure A.2 show identical windows.

Full frames are embedded as JPEG (quality 94, 4:4:4); the native crops stay lossless (Flate), and
the pixel audit checks each one byte for byte against a fresh crop of its source PNG.

Writes f5_qualitative_body.pdf and f5_qualitative_body_audit.json (crop manifest and pixel audit).
Regenerate: OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f5_qualitative_body.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import crope_style as cs
import f4_qualitative as fq            # same arms, crop rule, embedding and pixel audit as Figure A.2

ROWS = (8, 9)                          # the first two held-out manifest rows, fixed before viewing
TAGS = {8: "Stone bridge", 9: "Paris panorama"}
METHODS = fq.METHODS                   # NTK, DyPE, SEGA (released), SEGA base + temp., Constancy-TM, SEGA + TM
LABELS = {"sega_official": "SEGA\n(released)", "temp_only_sbase": "SEGA base\n+ temp."}  # two-line names
BOLD = fq.BOLD                         # the reference arm
FULL = fq.FULL

# ==== geometry (inches, from the top-left corner); built at the final print width, included at \linewidth
W = cs.TEXT                            # 5.50 in
FS = 8                                 # prompt tags and arm names (pt)
T = 0.74                               # side of every photo tile (float with a 4-line caption stays under 0.45 of the text block)
GR = 0.03                              # full frame to its crop
GI, GG = 0.035, 0.12                   # gap inside a column pair, gap between pairs
GP = 0.08                              # gap between the two prompt rows
BAND = 0.16                            # gray band behind each rotated prompt tag (A.2's tag band, turned)
BAND_GAP = 0.035                       # band to the grid
LAB_COL = BAND + BAND_GAP              # prompt-tag column
LAB_GAP = 0.05                         # grid bottom to the top of the arm names
LINE = FS * 1.2 / 72                   # leading of the two-line arm names
CAP = FS * 0.72 / 72                   # cap height of the sans face at FS
DESC = FS * 0.22 / 72                  # descender depth (the "y" of DyPE, the parentheses)
H = round(2 * (2 * T + GR) + GP + LAB_GAP + CAP + LINE + DESC + 0.01, 3)
OUT = cs.HERE / "f5_qualitative_body"
AUDIT = cs.HERE / "f5_qualitative_body_audit.json"


def col_x(j, x0):
    """Left edge of arm column j: pairs of columns, GI inside a pair and GG between pairs."""
    pair, k = divmod(j, 2)
    return x0 + pair * (2 * T + GI + GG) + k * (T + GI)


def draw(groups):
    grid_w = 6 * T + 3 * GI + 2 * GG
    margin = (W - LAB_COL - grid_w) / 2
    assert margin >= 0.02, f"grid does not fit: margin {margin:.3f} in"
    x0 = margin + LAB_COL
    fig = plt.figure(figsize=(W, H))
    for r, item in enumerate(groups):
        top = r * (2 * T + GR + GP)
        fig.add_artist(Rectangle((margin / W, 1 - (top + 2 * T + GR) / H), BAND / W, (2 * T + GR) / H,
                                 transform=fig.transFigure, fc=cs.PANEL, ec="none", zorder=0))
        fig.text((margin + BAND / 2) / W, 1 - (top + T + GR / 2) / H, TAGS[item["row"]], fontsize=FS,
                 color=cs.INK, rotation=90, ha="center", va="center_baseline")
        (box, color), = item["boxes"]
        for j, m in enumerate(METHODS):
            x = col_x(j, x0)
            ax = fq.tile(fig, H, x, top, T)
            ax.imshow(cs.thumb(item["src"][m], fq.THUMB), extent=(0, FULL, FULL, 0), interpolation="none")
            fq.zoom_box(ax, box, color)
            axc = fq.tile(fig, H, x, top + T + GR, T)
            axc.imshow(cs.crop(item["src"][m], box), interpolation="none")
            fq.crop_frame(axc, T, color)
    # Every line of an arm name sits on a fixed baseline (va="top" would follow each name's own ink, and
    # matplotlib anchors a multi-line "baseline" at its last line), so each line is its own text.
    first_baseline = 2 * (2 * T + GR) + GP + LAB_GAP + CAP
    for j, m in enumerate(METHODS):
        for k, line in enumerate(LABELS.get(m, cs.label(m)).split("\n")):
            fig.text((col_x(j, x0) + T / 2) / W, 1 - (first_baseline + k * LINE) / H, line, fontsize=FS,
                     color=cs.INK, ha="center", va="baseline", fontweight="bold" if m in BOLD else "normal")
    return fig, x0, margin


def check_label_room(fig, x0):
    """Arm names must not run into their neighbours: measure each one against its column pitch."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lines = {}
    for t in fig.texts:
        if t.get_rotation() == 0:
            bb = t.get_window_extent(r)
            lines.setdefault(round(t.get_position()[1], 6), []).append((bb.x0 / fig.dpi, bb.x1 / fig.dpi, t.get_text()))
    for y, spans in lines.items():
        spans.sort()
        gaps = [b[0] - a[1] for a, b in zip(spans, spans[1:])]
        print("arm-name gaps (in):", " ".join(f"{g:.3f}" for g in gaps))
        assert not gaps or min(gaps) > 0.04, spans
        assert spans[0][0] >= 0 and spans[-1][1] <= W, spans
    tiles_bottom = (2 * (2 * T + GR) + GP) * fig.dpi
    tops = [fig.bbox.height - t.get_window_extent(r).y1 for t in fig.texts if t.get_rotation() == 0]
    assert min(tops) > tiles_bottom + 0.02 * fig.dpi, "an arm name reaches into the photos"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cs.style()
    table, prompts = cs.heldout(), cs.prompts()
    groups = fq.prepare(ROWS, table, prompts, "common")
    recorded = {r["row"]: r["selected_box"] for r in json.loads(fq.MANIFEST.read_text())["main"]["rows"]}
    for g in groups:
        assert list(g["boxes"][0][0]) == recorded[g["row"]], (g["row"], g["boxes"], recorded[g["row"]])
    fig, x0, margin = draw(groups)
    check_label_room(fig, x0)
    print(f"figure size {W:.2f} x {H:.3f} in ({H / 9:.2f} of the 9 in text block); tile {T:.3f} in")
    cs.finish(fig, OUT, formats=("pdf",))      # figure audit; every image still lossless here
    pdf = Path(f"{OUT}.pdf")
    res = fq.embed_and_audit(pdf, {h for g in groups for h in g["sha"].values()}, "lossless-crops")
    checks = fq.audit_entries(groups, res, "body")
    audit = dict(
        schema="qualitative-body-v1",
        selection_rule="first two held-out manifest rows (8, 9), fixed before any output was viewed",
        crop_rule="common detail (review stage), identical boxes to Figure A.2 (f4_qualitative_main_crops.json)",
        layout="prompts as rows, arms as columns in palette pairs; full frame above its native crop",
        full_frame_thumbnails=f"{fq.THUMB}x{fq.THUMB} Lanczos reduction, embedded as JPEG quality {fq.JPEG_Q}, 4:4:4",
        crop_pixels="unaltered RGB native pixels, embedded losslessly (Flate)",
        size_in=[W, H], tile_in=T, grid_left_in=round(x0, 4), side_margin_in=round(margin, 4),
        include_width=r"\linewidth", rows=[g["receipt"] for g in groups],
        passed=all(c["native_pixels_embedded_unresampled"] and c["exact_in_final_pdf"]
                   and c["final_filter"] == "FlateDecode" for c in checks),
        criterion="every native crop is byte-exact and Flate (lossless) in the final PDF",
        pdf=str(pdf), pdf_sha256=sha(pdf), native_crops_checked=len(checks), checks=checks,
        generator_sha256=sha(__file__))
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"pixel audit passed={audit['passed']} ({len(checks)} crops)")
    assert audit["passed"]


if __name__ == "__main__":
    main()
