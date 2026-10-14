"""Exercise resumable output transactions without model weights or CUDA."""

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from crope.runtime import FluxExperiment
from crope.utils import image_seed, image_stem


class ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(axes_dims_rope=(4, 6, 6))
        self.pos_embed = torch.nn.Identity()
        self.attn_processors = {"toy.processor": object()}

    def set_attn_processor(self, processors):
        self.attn_processors = dict(processors)

    def forward(self, hidden_states, **kwargs):
        return (torch.ones_like(hidden_states) * 0.25,)


class ToyPipeline:
    def __init__(self):
        self.transformer = ToyTransformer()
        self.scheduler = SimpleNamespace(config={"shift": 1}, sigmas=None)
        self.vae = SimpleNamespace(enable_tiling=lambda: None)
        self._execution_device = "cpu"
        self.calls = 0

    def __call__(self, prompt, height, width, num_inference_steps, generator, output_type, **kwargs):
        self.calls += 1
        x = torch.randn(1, height // 16 * (width // 16), 64, generator=generator)
        self.scheduler.sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        for step in range(num_inference_steps):
            v = self.transformer(hidden_states=x, timestep=self.scheduler.sigmas[step:step + 1])[0]
            x = x + (self.scheduler.sigmas[step + 1] - self.scheduler.sigmas[step]) * v
        if output_type == "latent":
            return SimpleNamespace(images=x)
        color = int(x.mean().abs().item() * 100) % 255
        return SimpleNamespace(images=[Image.new("RGB", (width, height), (color, 20, 30))])


def test_seed_identity_is_method_independent_and_path_safe():
    assert image_seed("p", 2) == image_seed("p", 2)
    assert len({image_seed("p", 2), image_seed("q", 2), image_seed("p", 3)}) == 3
    assert "/" not in image_stem("../../a;b", 0)
    assert image_stem("a b", 0) != image_stem("a_b", 0)


def test_sample_resume_integrity_and_partial_recovery(tmp_path):
    pipe = ToyPipeline()
    original = pipe.transformer.pos_embed
    experiment = FluxExperiment(pipe, "ntk2", (64, 64))
    row = {"id": "p", "prompt": "A small test image."}
    record = experiment.sample(row, tmp_path, steps=3, record_power=True)
    assert pipe.calls == 1
    assert len(record["gains"]) == 3
    assert all(len(g["log_g"]) == 16 for g in record["gains"])
    experiment.sample(row, tmp_path, steps=3, record_power=True)
    assert pipe.calls == 1
    # A metadata interruption is retried. Preserve the old file as evidence.
    sidecar = tmp_path / f"{image_stem('p', 0)}.json"
    sidecar.rename(tmp_path / "interrupted_metadata.backup")
    retry = experiment.sample(row, tmp_path, steps=3, record_power=True)
    assert retry["png_sha256"] == record["png_sha256"]
    assert pipe.calls == 2
    with pytest.raises(ValueError, match="different settings"):
        experiment.sample(row, tmp_path, steps=4)
    experiment.close()
    assert pipe.transformer.pos_embed is original


def test_resume_rejects_corrupted_png(tmp_path):
    experiment = FluxExperiment(ToyPipeline(), "base", (64, 64))
    row = {"id": "p", "prompt": "A small test image."}
    record = experiment.sample(row, tmp_path, steps=2)
    Image.new("RGB", (64, 64), "red").save(record["image"])
    with pytest.raises(ValueError, match="checksum"):
        experiment.sample(row, tmp_path, steps=2)
    experiment.close()


def test_probe_gain_is_visible_in_record_and_fingerprint(tmp_path):
    experiment = FluxExperiment(ToyPipeline(), "ntk2", (64, 64),
                                probe_band=(0.125, 0.25), probe_factor=1.5)
    record = experiment.sample({"id": "probe", "prompt": "A test."}, tmp_path, steps=2)
    assert record["config"]["probe_band"] == (0.125, 0.25)
    expected = torch.tensor(1.5).log().item()
    for step in record["gains"]:
        assert any(abs(value - expected) < 1e-6 for value in step["log_g"])
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["complete"] and saved["seconds"] >= 0
    experiment.close()
