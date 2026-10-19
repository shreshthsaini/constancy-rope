"""F6: SEGA as printed in the paper versus the released code.

Regenerate:  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f6_sega_code.py

(a) Per-pair log logit gain over the 28 denoising steps, read from the JSON sidecars of the eight Krea dev
    prompts (out_krea/pilot/<method>/4096x4096/*.json, rows 0-7, seed 0), median over prompts. Rows are the
    28 rotary pairs of each spatial axis (dims 16-71 height, 72-127 width; both dims of a pair are equal),
    pair 0 = highest frequency. Separate released/paper colour scales are labelled explicitly.
    A common-scale line panel shows mean log gain per pair across all prompts and steps.
(b) Dev prompt 0 (the first manifest row, the prompt of Figure 5), seed 0, for the same four variants:
    Lanczos-reduced full frame and a native 384 x 256 px crop centred at (2048, 3072).
    Numbers: mean over the eight dev prompts with a 95% percentile bootstrap CI (2,000 resamples over prompts),
    from eval_krea/pilot/4096x4096/per_image.csv (CLIP score) and eval_krea/ir/pilot.csv (ImageReward).
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import crope_style as cs

S = cs.SCR
PID0 = "a4k_eval_4096_minus6873203881450547484"   # dev prompt 0
CROP = (2048 - 192, 3072 - 128, 2048 + 192, 3072 + 128)
# (method key, group, sub-label); groups share a header
VARIANTS = [("sega_official", "sega_official", "per-step"),
            ("sega_official_static", "sega_official", "static"),
            ("sega", "sega", "per-step"),
            ("sega_static", "sega", "static")]
HALO = [pe.withStroke(linewidth=1.8, foreground=cs.INK)]


def gains(method):
    files = sorted(glob.glob(str(S / "out_krea" / "pilot" / method / "4096x4096" / "*.json")))
    assert len(files) == 8, (method, len(files))
    order = cs.prompt_order()
    out, ids = [], []
    for f in files:
        j = json.load(open(f))
        c = j["config"]
        assert c["method"] == method and c["res"] == "4096x4096" and c["seed"] == 0 and order[c["prompt_id"]] < 8
        lg = np.array([g["log_g"] for g in j["gains"]])          # (28 steps, 128 dims)
        assert lg.shape == (28, 128) and np.allclose(lg[:, 0::2], lg[:, 1::2])
        out.append(lg[:, 16::2])                                  # 56 pairs: 28 height then 28 width
        ids.append(c["prompt_id"])
    return np.median(np.array(out), 0), ids, np.array(out)                       # (28 steps, 56 pairs)


def metrics():
    """Mean over the 8 dev prompts and the paired delta vs released SEGA with its 95% bootstrap CI."""
    d = pd.read_csv(S / "eval_krea" / "pilot" / "4096x4096" / "per_image.csv")
    ir = pd.read_csv(S / "eval_krea" / "ir" / "pilot.csv")
    keys = ["method", "prompt_id", "res"]
    d = d.merge(ir[keys + ["imagereward", "png"]], on=keys, how="inner", validate="one_to_one")
    assert (d.path == d.png).all(), "metrics and ImageReward scored different files"
    d = d[d.method.isin([v[0] for v in VARIANTS]) & (d.res == "4096x4096")]
    res = {}
    for m, _, _ in VARIANTS:
        a = d[d.method == m]
        assert len(a) == 8 and (a.seed == 0).all()
        res[m] = {k: (a[k].mean(), None if m == "sega_official" else cs.paired(d, m, k)[:4])
                  for k in ("clip", "imagereward")}
    return res


def num(fmt, x, sign=False):
    t = (("+" if sign and x > 0 else "") + fmt.format(x)).replace("-", "\u2212")
    return t


def add_ax(fig, x, top, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H])


def main():
    cs.style()
    G = {m: gains(m) for m, _, _ in VARIANTS}
    ids = G["sega_official"][1]
    assert all(set(G[m][1]) == set(ids) for m in G)
    M = metrics()
    vmax = np.ceil(max(np.abs(G[m][0]).max() for m in G) * 10) / 10
    released_max = np.ceil(max(np.abs(G[m][0]).max() for m in G if m.startswith("sega_official")) * 100) / 100
    norms = {"sega_official": Normalize(-released_max, released_max), "sega": Normalize(-vmax, vmax)}
    cmap = plt.get_cmap("RdBu_r")

    # ---- geometry (inches)
    W = cs.TEXT
    left = 0.50           # heatmap y labels / metric row labels
    right = 0.62          # colorbar and its label
    gin, gout = 0.05, 0.16
    w = (W - left - right - 2 * gin - gout) / 4
    xs = [left, left + w + gin, left + 2 * w + gin + gout, left + 3 * w + 2 * gin + gout]
    head = 0.42
    hb, hgap = 0.44, 0.10            # one heatmap block (28 pairs) and the gap between height and width
    hm_bot = head + 2 * hb + hgap
    profile_top = hm_bot + .62
    profile_h = .65
    img_top = profile_top + profile_h + .45
    crop_h = w * 256 / 384
    met_top = img_top + w + 0.03 + crop_h + 0.17
    LINE = 0.155
    H = met_top + 3 * LINE + 0.10
    fig = plt.figure(figsize=(W, H))

    # ---- group headers
    for gi, (m, sub) in enumerate((("sega_official", None), ("sega", None))):
        x0 = xs[2 * gi]
        fig.text((x0 + w + gin / 2) / W, 1 - 0.14 / H, cs.label(m), fontsize=7.5, fontweight="bold",
                 color=cs.text_color(m), ha="center", va="baseline")
        fig.add_artist(plt.Line2D([x0 / W, (x0 + 2 * w + gin) / W], [1 - 0.185 / H] * 2, lw=0.8,
                                  color=cs.color(m), transform=fig.transFigure, solid_capstyle="butt"))
    for (m, grp, sub), x0 in zip(VARIANTS, xs):
        fig.text(x0 / W, 1 - 0.32 / H, sub, fontsize=6.5, color=cs.INK2, ha="left", va="baseline")
        g = G[m][0]
        fig.text((x0 + w) / W, 1 - 0.32 / H, f"{g.min():.2f} to {g.max():.2f}".replace("-", "\u2212"),
                 fontsize=6, color=cs.INK2, ha="right", va="baseline")

    # ---- (a) heatmaps
    first = None
    for ci, ((m, grp, sub), x0) in enumerate(zip(VARIANTS, xs)):
        g = G[m][0]
        for bi, (name, sl) in enumerate((("Height", slice(0, 28)), ("Width", slice(28, 56)))):
            ax = add_ax(fig, x0, head + bi * (hb + hgap), w, hb)
            if first is None:
                first = ax
            else:
                ax.sharex(first)
                ax.sharey(first)
            ax.pcolormesh(np.arange(0.5, 29), np.arange(-0.5, 28), g[:, sl].T, cmap=cmap, norm=norms[grp],
                          shading="flat", edgecolors="face", linewidth=0.25, antialiased=True,
                          rasterized=False)   # face-colored edges close the hairline seams between cells
            ax.set_xlim(0.5, 28.5)
            ax.set_ylim(27.5, -0.5)
            ax.set_xticks([7, 14, 21, 28])
            ax.set_yticks([0, 27])
            ax.tick_params(length=2, pad=1.5, labelsize=6.5)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color(cs.RULE)
            if bi == 0:
                ax.tick_params(labelbottom=False, bottom=False)
            elif ci == 0:
                ax.set_xlabel("Denoising step", fontsize=7, labelpad=1.5)
            if ci == 0:
                ax.set_ylabel(name, fontsize=7, labelpad=1)
            else:
                ax.tick_params(labelleft=False, left=False)
            if bi == 1 and ci > 0:
                ax.set_xlabel("Denoising step", fontsize=7, labelpad=1.5)
    # Separate colour scales reveal the released gains; the line panel below uses one scale.
    for ci, (group, limit, label) in enumerate([
            ("sega_official", released_max, "Released"), ("sega", vmax, "Paper eqs.")]):
        cax = add_ax(fig, W - right + .10, head + ci * (hb + hgap), .07, hb)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norms[group], cmap=cmap), cax=cax)
        cb.set_ticks([-limit, 0, limit])
        cb.ax.tick_params(labelsize=6.2, length=2, width=.4, pad=1.5)
        cb.outline.set_linewidth(.4)
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cb.set_label(label, fontsize=6.2, labelpad=2)
    profile = add_ax(fig, left, profile_top, W - left - .08, profile_h)
    for m, group, sub in VARIANTS:
        values = G[m][2].mean(axis=(0, 1)).reshape(2, 28).mean(axis=0)
        profile.plot(np.arange(28), values, color=cs.color(group),
                     ls="--" if sub == "static" else "-", lw=1.1,
                     label=("Released" if group == "sega_official" else "Paper eqs.") + ", " + sub)
    profile.axhline(0, color=cs.RULE, lw=.6)
    profile.set_xlim(0, 27)
    profile.set_xticks([0, 9, 18, 27])
    profile.set_xlabel("Rotary pair (high to low frequency)", fontsize=6.8, labelpad=1)
    profile.set_ylabel(r"Mean $\log g$", fontsize=6.8, labelpad=2)
    profile.tick_params(labelsize=6.5, length=2, pad=1)
    profile.legend(loc="upper left", ncol=2, fontsize=6.3, frameon=False,
                   handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0, 1.4))
    fig.text(0.02 / W, 1 - (head + hb + hgap / 2) / H, "Rotary pair, high to low frequency", rotation=90,
             fontsize=6.5, color=cs.INK2, ha="left", va="center")

    # ---- (b) images, crops and numbers
    s = 768 / 4096
    for ci, ((m, grp, sub), x0) in enumerate(zip(VARIANTS, xs)):
        png = glob.glob(str(S / "out_krea" / "pilot" / m / "4096x4096" / f"{PID0}_*.png"))
        assert len(png) == 1
        side = json.load(open(png[0][:-4] + ".json"))
        assert side["config"]["prompt_id"] == PID0 and side["config"]["seed"] == 0
        ax = add_ax(fig, x0, img_top, w, w)
        ax.imshow(cs.thumb(png[0], 768), interpolation="none")
        ax.set_axis_off()
        r = Rectangle((CROP[0] * s, CROP[1] * s), (CROP[2] - CROP[0]) * s, (CROP[3] - CROP[1]) * s, fill=False,
                      lw=0.6, edgecolor="white")
        r.set_path_effects(HALO)
        ax.add_patch(r)
        cax2 = add_ax(fig, x0, img_top + w + 0.03, w, crop_h)
        cax2.imshow(cs.crop(png[0], CROP), interpolation="none")
        cax2.set_axis_off()
        cax2.add_patch(Rectangle((0, 0), 1, 1, transform=cax2.transAxes, fill=False, lw=0.4, edgecolor=cs.RULE,
                                 clip_on=False))
        for ri, key in enumerate(("clip", "imagereward")):
            fmt = "{:.%df}" % cs.METRIC_PRECISION[key]
            mean, delta = M[m][key]
            y = met_top + 2 * ri * LINE
            fig.text((x0 + w / 2) / W, 1 - y / H, num(fmt, mean), fontsize=7, color=cs.INK, ha="center",
                     va="baseline", fontweight="bold" if delta is None else "normal")
            if delta is None:
                txt = "reference"
            else:
                dm, lo, hi, n = delta
                assert n == 8
                txt = f"{num(fmt, dm, sign=True)}  [{num(fmt, lo)}, {num(fmt, hi)}]"
            sig_ = delta is not None and (delta[1] > 0 or delta[2] < 0)   # CI excludes zero: dark ink
            fig.text((x0 + w / 2) / W, 1 - (y + LINE * 0.92) / H, txt, fontsize=6, color=cs.INK if sig_ else cs.INK2,
                     ha="center", va="baseline")
    for top, h, name in ((img_top, w, "Full frame"), (img_top + w + 0.03, crop_h, "1:1 crop")):
        fig.text(0.30 / W, 1 - (top + h / 2) / H, name, rotation=90, fontsize=7, color=cs.INK, ha="center",
                 va="center")
    fig.text(0.13 / W, 1 - (img_top + w / 2) / H, "Dev prompt 0, seed 0", rotation=90, fontsize=6.5,
             color=cs.INK2, ha="center", va="center")
    for ri, name in enumerate(("CLIP score \u2191", "ImageReward \u2191")):
        y = met_top + 2 * ri * LINE
        fig.text(0.02 / W, 1 - y / H, name, fontsize=6.5, color=cs.INK, ha="left", va="baseline")
        fig.text(0.02 / W, 1 - (y + LINE * 0.92) / H, "\u0394 vs released", fontsize=6, color=cs.INK2,
                 ha="left", va="baseline")

    cs.panel_tag(fig, 0.0, 1 - 0.14 / H, "a")
    cs.panel_tag(fig, 0.0, 1 - (img_top - 0.10) / H, "b")
    cs.finish(fig, str(cs.HERE / "f6_sega_code"), photo=True, formats=("pdf", "svg"))
    for m in G:
        g = G[m][0]
        print(f"{m:22s} median log g range [{g.min():+.2f}, {g.max():+.2f}]  CLIP {M[m]['clip'][0]:.2f}"
              f"  IR {M[m]['imagereward'][0]:.3f}")


if __name__ == "__main__":
    main()
