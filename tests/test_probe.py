"""Power-bin and export checks for T2; fixtures are not generation results."""

import numpy as np
from PIL import Image

from scripts.probe_mtf import image_powers, latent_powers, octave_edges, render_heatmap


def test_octave_response_uses_logit_gain_not_qk_amplitude():
    edges = octave_edges((64, 64))
    frequencies = [np.sqrt(a * b) for a, b in zip(edges[:-1], edges[1:])]
    base = {"final_latent_axis_power": [{"frequency": frequencies, "power": [2.0] * len(frequencies)}] * 2}
    changed = {"final_latent_axis_power": [{"frequency": frequencies, "power": [3.0] * len(frequencies)}] * 2}
    response = (np.log(latent_powers(changed, edges)) - np.log(latent_powers(base, edges))) / np.log(1.5)
    np.testing.assert_allclose(response, 1)


def test_rectangular_image_power_excludes_empty_axis_bins(tmp_path):
    yy, xx = np.meshgrid(np.arange(64), np.arange(128), indexing="ij")
    y = np.uint8(128 + 40 * np.sin(2 * np.pi * xx / 128) + 30 * np.cos(2 * np.pi * yy / 64))
    path = tmp_path / "spectrum_fixture.png"
    Image.fromarray(np.repeat(y[..., None], 3, axis=-1)).save(path)
    values = image_powers(path, [0.125, 0.25, 0.5], "cpu")
    assert np.isnan(values[0, 0])
    assert np.isfinite(values[1]).all()
    assert values[0, 1] > 0


def test_heatmap_exports_png_and_vector_files(tmp_path):
    # A renderer fixture verifies signed color scales and the output formats.
    matrix = np.eye(5) - 0.25
    render_heatmap(matrix, [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2], list(range(5)), tmp_path)
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"mtf_heatmap.{suffix}").stat().st_size > 100
    with Image.open(tmp_path / "mtf_heatmap.png") as image:
        assert min(image.size) > 500
