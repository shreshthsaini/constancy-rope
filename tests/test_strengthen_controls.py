"""Independent checks of the factorial controls and fixed temperature sweep."""
import math

import pytest
import torch

from crope.gain import GainController
from crope.runtime import FluxExperiment
from test_runtime import ToyPipeline


@pytest.mark.parametrize('temperature,multiplier,expected', [
    ('none', None, 1.), ('sref', None, 4 ** .16),
    ('none', 1.1, 1.1), ('none', 4 / 3, 4 / 3), ('none', 1.45, 1.45)])
def test_temperature_is_absolute_logit_gain_on_every_axis(temperature, multiplier, expected):
    c = GainController('temp_only', grid=(256, 256), temperature=temperature,
                       temperature_multiplier=multiplier)
    c.update(torch.zeros(1, 256, 256), .5, x0hat=torch.zeros(1, 256, 256))
    torch.testing.assert_close(c.qk_scale.square(), torch.full((128,), expected))
    torch.testing.assert_close(c.log_g, torch.full((128,), math.log(expected)))
    assert not c.history[0].get('bins')


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_multiplier_rejected(value):
    with pytest.raises(ValueError, match='finite and positive'):
        GainController('temp_only', temperature='none', temperature_multiplier=value)


def test_multiplier_cannot_silently_modify_other_modes_or_named_temperature():
    for mode, temp in [('sega_official', 'none'), ('temp_only', 'sref')]:
        with pytest.raises(ValueError, match='requires temp_only'):
            GainController(mode, temperature=temp, temperature_multiplier=1.1)


def test_unique_evaluator_identity_preserves_controller_and_no_allocation(monkeypatch):
    monkeypatch.setenv('CROPE_ROPE', 'sega_official')
    e = FluxExperiment(ToyPipeline(), 'temp_only', (64, 64), temperature='none',
                       output_method='sega_base_none')
    try:
        c = e.config({'id': 'x', 'prompt': 'test'}, 0, 28, 4.5)
        assert c['method'] == 'sega_base_none' and c['controller_method'] == 'temp_only'
        assert c['rope']['method'] == 'sega_official' and e.official is None
    finally:
        e.close()


def test_schedule_default_retains_original_limit_at_init_and_generation():
    pipe = ToyPipeline()
    e = FluxExperiment(pipe, 'ntk2', (4096, 4096), schedule_mode='diffusers_default')
    try:
        assert pipe.scheduler.config['max_image_seq_len'] == 4096
        # The toy forward need not allocate a 4K latent to check the pipeline call boundary.
        class Observed(Exception):
            pass
        def call(self, **kwargs):
            assert self.scheduler.config['max_image_seq_len'] == 4096
            raise Observed
        original = ToyPipeline.__call__
        ToyPipeline.__call__ = call
        try:
            with pytest.raises(Observed):
                e.generate({'id': 'x', 'prompt': 'test'})
        finally:
            ToyPipeline.__call__ = original
        assert e.config({'id': 'x', 'prompt': 'test'}, 0, 28, 4.5)['schedule_mode'] == 'diffusers_default'
    finally:
        e.close()
