"""ImageReward scores for generated images, run with an environment whose transformers matches ImageReward 1.5
(the crope env has a newer transformers that breaks its BLIP code). Resumable: appends to a CSV keyed by png path.

usage: imagereward_score.py ROOT PROMPTS OUT_CSV [--device cuda]
ROOT holds METHOD/RES/*.png with the sidecar JSON written by scripts/sample.py (it carries prompt_id).
"""
import argparse, csv, glob, json, os
import torch
import ImageReward as RM
from huggingface_hub import hf_hub_download
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root"); p.add_argument("prompts"); p.add_argument("out_csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(a.prompts)}
    done = set()
    if os.path.exists(a.out_csv):
        done = {r["png"] for r in csv.DictReader(open(a.out_csv))}
    ck = hf_hub_download("THUDM/ImageReward", "ImageReward.pt", local_files_only=True)
    cf = hf_hub_download("THUDM/ImageReward", "med_config.json", local_files_only=True)
    model = RM.load(ck, device=a.device, med_config=cf)
    new = not os.path.exists(a.out_csv)
    with open(a.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["png", "method", "res", "prompt_id", "imagereward"])
        for png in sorted(glob.glob(os.path.join(a.root, "*", "*", "*.png"))):
            if png in done:
                continue
            meta = png[:-4] + ".json"
            if not os.path.exists(meta):
                continue
            pid = json.load(open(meta))["config"]["prompt_id"]
            method, res = png.split(os.sep)[-3], png.split(os.sep)[-2]
            with torch.no_grad():
                score = model.score(prompts[pid], Image.open(png).convert("RGB"))
            w.writerow([png, method, res, pid, f"{float(score):.6f}"]); f.flush()


if __name__ == "__main__":
    main()
