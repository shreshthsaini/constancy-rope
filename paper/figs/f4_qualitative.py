"""F5 (main text) and A.1 (appendix): qualitative 4K grids from the first held-out rows, seed 0.

Layout shared by both figures: methods are rows (NTK, DyPE, SEGA (released), SEGA base + temp.,
Constancy-TM, SEGA + TM) with their names at the left, and each prompt owns a column group under a
gray band that carries a 1 to 3 word tag (the full prompts go in the caption). Inside a group the
full frame (Lanczos-reduced, 1 pt zoom box) sits immediately left of the native crop (1 pt frame in
the box colour). With one tile of height per arm, a crop cannot sit below its full frame, so it
sits to the right.

Main figure (f4_qualitative_main.pdf): held-out rows 8, 9 and 10, one red crop per row at the
common-detail location of qual_helpers.common_detail_box (a review-stage rule). The full frames
are embedded as JPEG; the native crops stay lossless and the pixel audit checks them byte for byte.

Appendix figure (f4_qualitative.pdf): the same rows with the historical two-crop rule
(red: highest-detail tile of the across-arm mean luminance; cyan: exact image centre). Its 36
crops are placed unresampled and then embedded as JPEG (lossless would exceed 11 MB); the audit
records their fidelity against the native pixels.

Writes f4_qualitative_main.pdf, f4_qualitative.pdf, f4_qualitative_main_crops.json and
f4_qualitative_main_pixel_audit.json. Usage: python f4_qualitative.py [--only main|appendix]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

import crope_style as cs
import qual_helpers as qh

ROWS = (8, 9, 10)                      # the first three held-out rows (held-out starts at row 8)
TAGS = {8: "Stone bridge", 9: "Paris panorama", 10: "Artist’s corner"}
OPENING = {8: "A stone bridge with arched spans", 9: "A panoramic view of Paris",
           10: "A cluttered artist's corner"}            # guards the row-to-prompt mapping
METHODS = ["ntk2", "dype", "sega_official", "temp_only_sbase", "constancy_tm_sbase", "sega_ctm_v2"]
BOLD = {"sega_official"}               # the reference arm
CROP = 384                             # native crop edge (source pixels)
THUMB = 384                            # full-frame reduction (Lanczos)
FULL = 4096

W = cs.TEXT                            # 5.50 in, included at \linewidth
FS = 8                                 # row labels and prompt tags (pt)
GI, GG = 0.03, 0.10                    # gutter inside a prompt group, gap between groups (in)
GR, GP = 0.03, 0.07                    # gap inside a method pair, gap between pairs (in)
BAND, BAND_GAP = 0.16, 0.035           # tag band height, band to first row (in)
TAG_PAD = 0.05                         # tag text inset from the band edge (in)
PAD = 0.07                             # label right edge to the grid (in)
MIN_MARGIN = 0.03                      # side margin of the width-limited appendix grid (in)
LW = 1.0                               # zoom box and crop frame (pt)
MAIN_H = 4.30                          # height cap of the main figure (in)
JPEG_Q = 94                            # JPEG quality (4:4:4) for embedded photos, as crope_style.jpegify

# The methods-as-columns appendix grid this script replaced (same crop rule, same rows).
PREVIOUS_APPENDIX = dict(
    sha256="0a49346b797be25f709e064cabd6b6b92342395603fb3a9ca30bff0394ff25a8",
    original_generator_sha256="87668e4884a955500ca3b604b5a1922a77b932d39b9dcbaa679065bda3e48a20",
    original_helper_sha256="685413fa49ef123cacc08a964a15df2d3403ed95169de34aaef4d33eb2a9efa7",
    source_git_commit="4bf534d16b821f45355b6e5354cfd1c6bdbe0448")

OUT_MAIN = cs.HERE / "f4_qualitative_main"
OUT_APP = cs.HERE / "f4_qualitative"
MANIFEST = cs.HERE / "f4_qualitative_main_crops.json"
AUDIT = cs.HERE / "f4_qualitative_main_pixel_audit.json"


# ---------------------------------------------------------------- layout
def label_width():
    """Widest method label at FS pt (inches), measured with the figure font."""
    fig = plt.figure(figsize=(W, 1))
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for m in METHODS:
        t = fig.text(0, 0, cs.label(m), fontsize=FS, fontweight="bold" if m in BOLD else "normal")
        widest = max(widest, t.get_window_extent(renderer).width / fig.dpi)
    plt.close(fig)
    return widest


def grid_height(t):
    return BAND + BAND_GAP + 6 * t + 3 * GR + 2 * GP


def row_top(i, t):
    """Distance from the figure top to the top of method row i (rows grouped in pairs)."""
    pair, k = divmod(i, 2)
    return BAND + BAND_GAP + pair * (2 * t + GR + GP) + k * (t + GR)


def group_width(ncols, t):
    return ncols * t + (ncols - 1) * GI


def tile(fig, H, x, top, t):
    ax = fig.add_axes([x / W, (H - top - t) / H, t / W, t / H])
    ax.set_axis_off()
    return ax


def zoom_box(ax, box, color):
    """1 pt square outline on a full frame shown with extent (0, 4096, 4096, 0)."""
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=LW,
                           joinstyle="miter", zorder=5))


def crop_frame(ax, t, color):
    """1 pt frame drawn just inside the crop edge, so tiles keep their exact outline."""
    inset = (LW / 2 / 72) / t
    ax.add_patch(Rectangle((inset, inset), 1 - 2 * inset, 1 - 2 * inset, transform=ax.transAxes,
                           fill=False, ec=color, lw=LW, joinstyle="miter", zorder=5))


def draw(groups, t, H, ncols):
    """groups: per prompt dict(row, src, boxes=[(box, colour), ...]); returns (fig, x0)."""
    lab_w = label_width()
    gw = group_width(ncols, t)
    grid_w = len(groups) * gw + (len(groups) - 1) * GG
    margin = (W - lab_w - PAD - grid_w) / 2
    assert margin >= -1e-9, f"grid does not fit: margin {margin:.3f} in"
    x0 = margin + lab_w + PAD
    fig = plt.figure(figsize=(W, H))
    for i, m in enumerate(METHODS):
        fig.text((x0 - PAD) / W, (H - row_top(i, t) - t / 2) / H, cs.label(m), fontsize=FS,
                 color=cs.INK, ha="right", va="center_baseline",
                 fontweight="bold" if m in BOLD else "normal")
    for g, item in enumerate(groups):
        gx = x0 + g * (gw + GG)
        fig.add_artist(Rectangle((gx / W, (H - BAND) / H), gw / W, BAND / H, transform=fig.transFigure,
                                 fc=cs.PANEL, ec="none", zorder=0))
        fig.text((gx + TAG_PAD) / W, (H - BAND / 2) / H, TAGS[item["row"]], fontsize=FS, color=cs.INK,
                 ha="left", va="center_baseline")
        for i, m in enumerate(METHODS):
            top = row_top(i, t)
            ax = tile(fig, H, gx, top, t)
            ax.imshow(cs.thumb(item["src"][m], THUMB), extent=(0, FULL, FULL, 0), interpolation="none")
            for box, color in item["boxes"]:
                zoom_box(ax, box, color)
            for k, (box, color) in enumerate(item["boxes"], start=1):
                axc = tile(fig, H, gx + k * (t + GI), top, t)
                axc.imshow(cs.crop(item["src"][m], box), interpolation="none")
                crop_frame(axc, t, color)
    return fig, x0, margin


# ---------------------------------------------------------------- native-crop audit
def native_crop_sha(path, boxes):
    """sha256 of the RGB bytes of each crop cut fresh from the source PNG (no cache)."""
    out = []
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        for box in boxes:
            a = np.asarray(rgb.crop(tuple(box)), dtype=np.uint8)
            assert a.shape == (CROP, CROP, 3)
            out.append(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    return out


_PDF_SNIPPET = r"""
import hashlib, io, json, math, os, sys
import fitz
from PIL import Image, ImageChops, ImageFile, ImageStat
ImageFile.MAXBLOCK = 1 << 26
src, quality, spec_path, out_path, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
keep = set(json.load(open(spec_path)))

def hashes(pix):
    rows = [bytes(pix.samples[r * pix.stride:(r + 1) * pix.stride]) for r in range(pix.height)]
    return hashlib.sha256(b"".join(rows)).hexdigest(), hashlib.sha256(b"".join(rows[::-1])).hexdigest()

doc = fitz.open(src); crops, n_jpeg = {}, 0
for page in doc:
    for info in page.get_images(full=True):
        xref, smask = info[0], info[1]
        pix = fitz.Pixmap(doc, xref)
        if smask or pix.alpha or pix.n != 3:
            continue
        stored, flipped = hashes(pix)
        h = stored if stored in keep else flipped if flipped in keep else None
        if h is not None:
            crops[h] = dict(width=pix.width, height=pix.height, exact_before_embedding=True,
                            storage_order="as stored" if h == stored else "rows flipped")
            if mode == "lossless-crops":
                continue
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
        if h is not None:
            diff = ImageChops.difference(im, Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
            mse = sum(ImageStat.Stat(diff).sum2) / (3 * pix.width * pix.height)
            crops[h].update(jpeg_psnr_db=round(10 * math.log10(255 ** 2 / mse), 2) if mse else None,
                            jpeg_max_abs_error=max(e[1] for e in diff.getextrema()))
        page.replace_image(xref, stream=buf.getvalue()); n_jpeg += 1
tmp = src + ".tmp.pdf"; doc.save(tmp, garbage=4, deflate=True); doc.close(); os.replace(tmp, src)

doc = fitz.open(src); filters, seen = {}, set()
for page in doc:
    for info in page.get_images(full=True):
        if info[0] in seen:                     # replace_image leaves a second resource name per image
            continue
        seen.add(info[0])
        filters[info[8]] = filters.get(info[8], 0) + 1
        pix = fitz.Pixmap(doc, info[0])
        if pix.n != 3:
            continue
        for h in hashes(pix):
            if h in crops:
                crops[h].update(exact_in_final_pdf=True, final_filter=info[8])
for c in crops.values():
    c.setdefault("exact_in_final_pdf", False)
json.dump(dict(jpeg_embedded=n_jpeg, filters=filters, crops=crops, bytes=os.path.getsize(src)),
          open(out_path, "w"))
"""


def embed_and_audit(pdf, crop_hashes, mode, quality=JPEG_Q):
    """JPEG-embed the full frames (and, in mode "jpeg-all", the crops), then audit every crop.

    Every crop is first found byte for byte in matplotlib's output (native pixels, no resampling).
    Mode "lossless-crops" keeps them that way and re-finds them in the final file; mode "jpeg-all"
    re-encodes them and records the JPEG fidelity against the native pixels."""
    env = dict(os.environ, PYTHONPATH=cs._PDFLIB, OMP_NUM_THREADS="8")
    with tempfile.TemporaryDirectory(dir=cs.CACHE) as tmp:
        spec, out = Path(tmp) / "keep.json", Path(tmp) / "out.json"
        spec.write_text(json.dumps(sorted(crop_hashes)))
        r = subprocess.run([cs._PDFPY, "-s", "-c", _PDF_SNIPPET, str(pdf), str(quality), str(spec), str(out), mode],
                           env=env, capture_output=True, text=True)
        sys.stderr.write(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError("PDF embedding or audit failed")
        res = json.loads(out.read_text())
    exact = sum(c["exact_in_final_pdf"] for c in res["crops"].values())
    print(f"{Path(pdf).name}: {res['jpeg_embedded']} images as JPEG q{quality}, "
          f"{len(res['crops'])}/{len(crop_hashes)} native crops found before embedding, "
          f"{exact} byte-exact in the final file, filters {res['filters']}, {res['bytes'] / 1e6:.2f} MB",
          file=sys.stderr)
    return res


def audit_entries(groups, res, which):
    checks = []
    for item in groups:
        for m in METHODS:
            for k, (box, color) in enumerate(item["boxes"]):
                h = item["sha"][(m, k)]
                hit = res["crops"].get(h, {})
                checks.append(dict(figure=which, row=item["row"], method=m, crop=["red", "cyan"][k],
                                   box=list(box), native_rgb_sha256=h,
                                   native_pixels_embedded_unresampled=bool(hit.get("exact_before_embedding")),
                                   **{k2: v for k2, v in hit.items() if k2 != "exact_before_embedding"}))
    return checks


# ---------------------------------------------------------------- figures
def prepare(rows, table, prompts, rule):
    groups = []
    for row in rows:
        assert prompts[row]["prompt"].startswith(OPENING[row]), row
        src = qh.sources(row, METHODS, table=table)
        paths = [src[m] for m in METHODS]
        if rule == "common":
            box, receipt = qh.common_detail_box(paths, CROP)
            boxes = [(tuple(box), cs.ZOOM)]
            print(f"main row {row}: {box}; minimum normalized energy "
                  f"{receipt['minimum_normalized_energy']:.6f}", file=sys.stderr)
        else:
            box_b = qh.center_box(CROP)
            box_a = qh.detail_box(qh.mean_luma(paths), CROP, exclude=[box_b])
            boxes = [(tuple(box_a), cs.ZOOM), (tuple(box_b), cs.ZOOM2)]
            receipt = dict(rule_red="maximum mean squared Laplacian of the across-arm mean luminance, "
                                    "stride 96, margin 64, excluding tiles that overlap the centre box",
                           rule_cyan="exact image centre", red_box=list(box_a), cyan_box=list(box_b))
            print(f"appendix row {row}: red {box_a}  cyan {box_b}", file=sys.stderr)
        sha = {}
        for m in METHODS:
            for k, h in enumerate(native_crop_sha(src[m], [b for b, _ in boxes])):
                sha[(m, k)] = h
        receipt.update(row=row, prompt_id=prompts[row]["id"], prompt=prompts[row]["prompt"], tag=TAGS[row],
                       methods=METHODS, seed=0,
                       source_sha256={m: hashlib.sha256(src[m].read_bytes()).hexdigest() for m in METHODS})
        groups.append(dict(row=row, src=src, boxes=boxes, sha=sha, receipt=receipt))
    return groups


def build(groups, t, H, ncols, stem, mode):
    fig, x0, margin = draw(groups, t, H, ncols)
    cs.finish(fig, stem, formats=("pdf",))      # figure audit; every image still lossless here
    pdf = Path(f"{stem}.pdf")
    res = embed_and_audit(pdf, {h for g in groups for h in g["sha"].values()}, mode)
    return pdf, res, dict(tile_in=float(t), grid_left_in=round(x0, 4), side_margin_in=round(margin, 4),
                          size_in=[W, round(H, 4)], include_width=r"\linewidth")


def main_figure(table, prompts):
    """Height-limited: six rows of tiles fill MAIN_H; three prompt pairs then fill the width."""
    t = np.floor((MAIN_H - grid_height(0)) / 6 * 1000) / 1000
    groups = prepare(ROWS, table, prompts, "common")
    return (groups, *build(groups, t, grid_height(t), 2, OUT_MAIN, "lossless-crops"))


def appendix_figure(table, prompts):
    """Width-limited: nine columns (full, red crop, cyan crop per prompt) fill the text width."""
    free = W - 2 * MIN_MARGIN - label_width() - PAD - len(ROWS) * 2 * GI - (len(ROWS) - 1) * GG
    t = np.floor(free / (3 * len(ROWS)) * 1000) / 1000
    groups = prepare(ROWS, table, prompts, "legacy")
    return (groups, *build(groups, t, grid_height(t), 3, OUT_APP, "jpeg-all"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("main", "appendix"), help="build one figure only")
    args = parser.parse_args()
    cs.style()
    table, prompts = cs.heldout(), cs.prompts()
    old_manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old_audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    manifest = dict(schema="qualitative-grids-v3", method_order=METHODS,
                    layout="methods as rows, labels at left; per prompt a full frame with its native "
                           "crop(s) immediately to the right; tag band above each prompt group",
                    full_frame_thumbnails=f"{THUMB}x{THUMB} Lanczos reduction of the original full frame, "
                                          f"embedded as JPEG quality {JPEG_Q}, 4:4:4")
    reference = "each crop cut fresh from its source PNG (no cache) and hashed as RGB bytes"
    audit = old_audit if old_audit.get("schema") == "qualitative-pixel-audit-v3" else {}
    audit["schema"] = "qualitative-pixel-audit-v3"
    if args.only in (None, "main"):
        groups, pdf, res, geo = main_figure(table, prompts)
        checks = audit_entries(groups, res, "main")
        manifest["main"] = dict(path=str(pdf), sha256=sha(pdf), crop_rule="common detail (review stage)",
                                crop_pixels="unaltered RGB native pixels, embedded losslessly (Flate)",
                                rows=[g["receipt"] for g in groups], **geo)
        audit["main"] = dict(
            passed=all(c["native_pixels_embedded_unresampled"] and c["exact_in_final_pdf"]
                       and c["final_filter"] == "FlateDecode" for c in checks),
            criterion="every native crop is byte-exact and Flate (lossless) in the final PDF",
            pdf=str(pdf), pdf_sha256=sha(pdf), native_crops_checked=len(checks), reference=reference,
            checks=checks)
    if args.only in (None, "appendix"):
        groups, pdf, res, geo = appendix_figure(table, prompts)
        checks = audit_entries(groups, res, "appendix")
        manifest["appendix"] = dict(path=str(pdf), sha256=sha(pdf), crop_rule="historical two-crop rule",
                                    crop_pixels=f"native pixels, never resampled; embedded as JPEG quality "
                                                f"{JPEG_Q}, 4:4:4 (lossless would exceed 11 MB)",
                                    rows=[g["receipt"] for g in groups], previous_version=PREVIOUS_APPENDIX,
                                    **geo)
        psnr = [c["jpeg_psnr_db"] for c in checks if c.get("jpeg_psnr_db")]
        audit["appendix"] = dict(
            passed=all(c["native_pixels_embedded_unresampled"] for c in checks),
            criterion="every native crop is placed unresampled (byte-exact before JPEG embedding); "
                      "JPEG fidelity against the native pixels is recorded per crop",
            jpeg_psnr_db_min=min(psnr) if psnr else None, jpeg_psnr_db_median=float(np.median(psnr)) if psnr else None,
            pdf=str(pdf), pdf_sha256=sha(pdf), native_crops_checked=len(checks), reference=reference,
            checks=checks)
    for key in ("main", "appendix"):
        manifest.setdefault(key, old_manifest.get(key))
    audit["passed"] = bool((audit.get("main") or {}).get("passed"))   # this file's headline: the main figure
    manifest.update(generator_sha256=sha(__file__), helper_sha256=sha(qh.__file__))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"pixel audit: main passed={(audit.get('main') or {}).get('passed')}, "
          f"appendix passed={(audit.get('appendix') or {}).get('passed')}", file=sys.stderr)


if __name__ == "__main__":
    main()
