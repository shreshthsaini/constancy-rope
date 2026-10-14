"""Adapters use official scoring APIs; import and load only one at a time.

Sources: https://huggingface.co/yuvalkirstain/PickScore_v1
https://github.com/zai-org/ImageReward and https://github.com/MizzenAI/HPSv3
https://github.com/chaofengc/IQA-PyTorch and https://github.com/toshas/torch-fidelity
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .image import crop_boxes


def tensor_rgb(image: Image.Image, *, uint8: bool = False):
    import torch

    tensor = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).unsqueeze(0)
    return tensor if uint8 else tensor.float() / 255.0


class ClipScorer:
    def __init__(self, metric: str, device: str):
        from transformers import AutoModel, AutoProcessor

        self.metric, self.device = metric, device
        model = "openai/clip-vit-large-patch14" if metric == "clip" else "yuvalkirstain/PickScore_v1"
        processor = model if metric == "clip" else "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.processor = AutoProcessor.from_pretrained(processor, local_files_only=True)
        self.model = AutoModel.from_pretrained(model, local_files_only=True).eval().to(device)
        self.provenance = {"model": model, "processor": processor,
                           "revision": getattr(self.model.config, "_commit_hash", None),
                           "definition": "100*max(cosine,0)" if metric == "clip" else "exp(logit_scale)*cosine"}

    def score(self, record: dict, **kwargs) -> dict:
        import torch
        import torch.nn.functional as functional

        with Image.open(record["path"]) as image:
            image_inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=[record["prompt"]], padding=True, truncation=True,
                                     max_length=77, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            # Transformers 5 may return a pooled output object for these methods.
            if not torch.is_tensor(image_features):
                image_features = image_features.pooler_output
            if not torch.is_tensor(text_features):
                text_features = text_features.pooler_output
            cosine = (functional.normalize(image_features, dim=-1) * functional.normalize(text_features, dim=-1)).sum()
            score = 100 * cosine.clamp(min=0) if self.metric == "clip" else self.model.logit_scale.exp() * cosine
        return {self.metric: float(score.item())}


class ImageRewardScorer:
    def __init__(self, device: str):
        import ImageReward as reward
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download("THUDM/ImageReward", "ImageReward.pt", local_files_only=True)
        config = hf_hub_download("THUDM/ImageReward", "med_config.json", local_files_only=True)
        self.model = reward.load(checkpoint, device=device, med_config=config)
        self.provenance = {"model": "THUDM/ImageReward", "checkpoint": checkpoint,
                           "definition": "official normalized ImageReward-v1.0 score"}

    def score(self, record: dict, **kwargs) -> dict:
        import torch

        with torch.inference_mode():
            value = self.model.score(record["prompt"], record["path"])
        return {"imagereward": float(value)}


class HPSv3Scorer:
    def __init__(self, device: str):
        from huggingface_hub import hf_hub_download, snapshot_download

        if importlib.util.find_spec("hpsv3") is None:
            raise RuntimeError("HPSv3 package is unavailable; install its official dependencies into crope")
        base_id = "Qwen/Qwen2-VL-7B-Instruct"
        try:
            base = Path(snapshot_download(base_id, local_files_only=True))
            index = base / "model.safetensors.index.json"
            if index.exists():
                required = set(json.loads(index.read_text())["weight_map"].values())
                if not all((base / name).is_file() for name in required):
                    raise FileNotFoundError("Incomplete base model shards")
            elif not (base / "model.safetensors").is_file():
                raise FileNotFoundError("No complete base checkpoint")
        except Exception as exc:
            raise RuntimeError("HPSv3 also needs the complete cached Qwen2-VL-7B-Instruct base; "
                               "this evaluator will not download that extra large model") from exc
        checkpoint = hf_hub_download("MizzenAI/HPSv3", "HPSv3.safetensors", local_files_only=True)
        from hpsv3 import HPSv3RewardInferencer

        self.model = HPSv3RewardInferencer(checkpoint_path=checkpoint, device=device)
        self.provenance = {"model": "MizzenAI/HPSv3", "base_model": base_id,
                           "checkpoint": checkpoint, "definition": "official reward mu, not sigma"}

    def score(self, record: dict, **kwargs) -> dict:
        value = self.model.reward(prompts=[record["prompt"]], image_paths=[record["path"]])
        return {"hpsv3": float(value[0][0].item())}


class PyiqaScorer:
    def __init__(self, metric: str, device: str):
        import pyiqa

        self.metric, self.device = metric, device
        self.model = pyiqa.create_metric(metric, device=device).eval()
        self.provenance = {"model": f"pyiqa/{metric}", "full_input": "native resolution, model preprocessing unchanged",
                           "crop_input": "random 1024x1024, averaged within image"}

    def score(self, record: dict, *, crop_seed: int, crops: int, **kwargs) -> dict:
        import torch

        with Image.open(record["path"]) as source, torch.inference_mode():
            image = source.convert("RGB")
            full = float(self.model(tensor_rgb(image).to(self.device)).mean().item())
            boxes = crop_boxes(image.size, record["prompt_id"], seed=crop_seed, count=crops)
            crop_values = [float(self.model(tensor_rgb(image.crop(box)).to(self.device)).mean().item()) for box in boxes]
        return {self.metric: full, self.metric + "_crop1024": float(np.mean(crop_values))}


def load_scorer(metric: str, device: str):
    if metric in ("clip", "pickscore"):
        return ClipScorer(metric, device)
    if metric == "imagereward":
        return ImageRewardScorer(device)
    if metric == "hpsv3":
        return HPSv3Scorer(device)
    if metric in ("musiq", "clipiqa"):
        return PyiqaScorer(metric, device)
    raise ValueError(f"Unknown learned metric: {metric}")


class InceptionFeatures:
    """Canonical FID Inception weights, never ImageNet classification weights."""

    def __init__(self, device: str):
        self.device = device
        if importlib.util.find_spec("torch_fidelity") is not None:
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

            self.model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).eval().to(device)
            self.uint8 = True
            self.provenance = {"backend": "torch-fidelity", "feature_dim": 2048,
                               "resize": "official TensorFlow-compatible bilinear to 299x299"}
        elif importlib.util.find_spec("pytorch_fid") is not None:
            from pytorch_fid.inception import InceptionV3

            self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], use_fid_inception=True).eval().to(device)
            self.uint8 = False
            self.provenance = {"backend": "pytorch-fid", "feature_dim": 2048,
                               "resize": "official PyTorch bilinear to 299x299"}
        else:
            raise RuntimeError("FID/KID require torch-fidelity (preferred) or pytorch-fid in crope")

    def one(self, image: Image.Image) -> np.ndarray:
        import torch

        with torch.inference_mode():
            value = self.model(tensor_rgb(image, uint8=self.uint8).to(self.device))[0]
        return value.flatten(1).float().cpu().numpy()

    def extract(self, record: dict, *, patch: bool, crop_seed: int, crops: int) -> np.ndarray:
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")
            if not patch:
                return self.one(image)
            boxes = crop_boxes(image.size, record["prompt_id"], seed=crop_seed, count=crops)
            return np.concatenate([self.one(image.crop(box)) for box in boxes], axis=0)
