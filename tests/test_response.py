"""A stateful mock sampler checks that probes cannot alter the baseline trajectory."""
from types import SimpleNamespace

import pytest
import torch

from crope.response import Direction, SameStateProbe, band_directions, perturb_gain
from crope.runtime import FluxExperiment


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(axes_dims_rope=(4, 8, 8))
        self.pos_embed = torch.nn.Identity()
        self.x_embedder = torch.nn.Identity()
        self.attn_processors = {'mock.processor': object()}
        self.controller = None
        self.seen = []
        self.fail_call = None

    def set_attn_processor(self, processors):
        self.attn_processors = dict(processors)

    def forward(self, hidden_states, **kwargs):
        x = self.x_embedder(hidden_states)
        g = self.controller.qk_scale
        self.seen.append((hidden_states.clone(), g.clone()))
        if self.fail_call == len(self.seen):
            raise RuntimeError('injected probe failure')
        weights = torch.arange(1, len(g) + 1, dtype=g.dtype)
        # Deliberate randomness exercises RNG restoration even in eval mode.
        result = x * (g @ weights) / weights.sum() + .01 * torch.rand_like(x)
        return (result,)


def make_experiment():
    transformer = Transformer().eval()
    config = {}
    pipe = SimpleNamespace(transformer=transformer, scheduler=SimpleNamespace(
        config=config, sigmas=torch.tensor([.8, .4, 0.]),
        register_to_config=lambda **kw: config.update(kw)))
    experiment = FluxExperiment(pipe, 'sega_official', (128, 128))
    transformer.eval()
    transformer.controller = experiment.controller
    return experiment


def direction():
    vector = torch.zeros(20)
    vector[4:8] = 1
    return Direction('height_low', vector, (1,))


def rollout(experiment, x):
    for index in range(2):
        output = experiment.pipe.transformer(hidden_states=x, timestep=torch.tensor([.8 - .4 * index]))[0]
        x = x - .4 * output
    return x


def test_rms_control_preserves_axis_energy_pairs_and_other_axes():
    base = torch.arange(1., 11.).repeat_interleave(2)
    changed = perturb_gain(base, direction(), .2, (4, 8, 8))
    torch.testing.assert_close(changed[4:12].square().sum(), base[4:12].square().sum())
    assert torch.equal(changed[:4], base[:4]) and torch.equal(changed[12:], base[12:])
    assert torch.equal(changed[::2], changed[1::2])
    assert torch.equal(perturb_gain(base, direction(), 0, (4, 8, 8)), base)


def test_scalar_control_has_declared_logit_amplitude():
    vector = torch.ones(20); vector[:4] = 0
    base = torch.ones(20)
    changed = perturb_gain(base, Direction('scalar', vector), .2, (4, 8, 8))
    torch.testing.assert_close(2 * changed[4:].log(), torch.full((16,), .2))


def test_index_axis_and_nonpaired_directions_rejected():
    for index in (0, 4):
        vector = torch.zeros(20); vector[index] = 1
        with pytest.raises(ValueError, match='index axis and rotary pairs'):
            perturb_gain(torch.ones(20), Direction('invalid', vector), .1, (4, 8, 8))


@torch.inference_mode()
def test_trajectory_rng_history_and_sampler_state_match_unprobed_run():
    x = torch.randn(1, 64, 4, generator=torch.Generator().manual_seed(5))
    baseline = make_experiment()
    torch.manual_seed(17)
    expected = rollout(baseline, x.clone())
    rng_expected = torch.get_rng_state().clone()
    history_expected = baseline.controller.history
    probe_exp = make_experiment()
    torch.manual_seed(17)
    original_forward = probe_exp.pipe.transformer.forward
    history = probe_exp.controller.history
    with SameStateProbe(probe_exp, [0, 1], [direction()]) as probe:
        actual = rollout(probe_exp, x.clone())
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng_expected)
    assert probe_exp.controller.history is history
    assert history == history_expected
    assert probe_exp.step == baseline.step == 2
    assert probe.extra_calls == 10
    assert len(probe.records) == 2
    assert all(record['replay_max_abs_velocity_error'] == 0 for record in probe.records)
    assert probe_exp.pipe.transformer.forward == original_forward
    assert 'forward' not in probe_exp.pipe.transformer.__dict__
    # Each baseline call and all five replays see exactly the same x_t.
    seen = probe_exp.pipe.transformer.seen
    for start in (0, 6):
        assert all(torch.equal(seen[start][0], value[0]) for value in seen[start:start + 6])
    torch.testing.assert_close(probe_exp.previous_x0, baseline.previous_x0, rtol=0, atol=0)
    assert probe_exp.official.diagnostics == baseline.official.diagnostics
    assert len(probe_exp.pipe.transformer.x_embedder._forward_hooks) == 1
    baseline.close(); probe_exp.close()


@torch.inference_mode()
def test_probe_failure_restores_controller_hooks_and_rng():
    exp = make_experiment()
    exp.pipe.transformer.fail_call = 3  # baseline, replay, first intervention
    x = torch.randn(1, 64, 4)
    history = exp.controller.history
    reference = make_experiment()
    torch.manual_seed(91)
    reference.pipe.transformer(hidden_states=x, timestep=torch.tensor([.8]))
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    with pytest.raises(RuntimeError, match='injected'):
        with SameStateProbe(exp, [0], [direction()]):
            exp.pipe.transformer(hidden_states=x, timestep=torch.tensor([.8]))
    assert exp.controller.history is history
    assert history == reference.controller.history
    assert torch.equal(exp.controller.qk_scale, reference.controller.qk_scale)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert 'forward' not in exp.pipe.transformer.__dict__
    assert len(exp.pipe.transformer.x_embedder._forward_hooks) == 1
    exp.close(); reference.close()


def test_generated_controls_are_separate_axes_and_include_scalar():
    exp = make_experiment()
    directions = band_directions(exp)
    assert directions[-1].name == 'spatial_scalar'
    for d in directions[:-1]:
        assert len(d.preserve_axes) == 1
        axis = d.preserve_axes[0]
        low, high = sum(exp.axes_dims[:axis]), sum(exp.axes_dims[:axis + 1])
        assert d.vector[:low].count_nonzero() == 0
        assert d.vector[high:].count_nonzero() == 0
    exp.close()


def test_training_mode_and_nested_wrappers_rejected():
    exp = make_experiment()
    exp.pipe.transformer.train()
    with pytest.raises(ValueError, match='eval-mode'):
        SameStateProbe(exp, [0])
    exp.pipe.transformer.eval()
    with SameStateProbe(exp, [0]):
        with pytest.raises(ValueError, match='nested'):
            with SameStateProbe(exp, [0]):
                pass
    exp.close()


@torch.inference_mode()
def test_real_diffusers_attention_keeps_cpu_trajectory_exact():
    """Exercise actual FLUX blocks and processors, using tiny random weights."""
    from diffusers import FluxTransformer2DModel
    torch.manual_seed(712)
    transformer = FluxTransformer2DModel(
        patch_size=1, in_channels=8, out_channels=8, num_layers=1,
        num_single_layers=1, attention_head_dim=20, num_attention_heads=1,
        joint_attention_dim=12, pooled_projection_dim=6, guidance_embeds=True,
        axes_dims_rope=(4, 8, 8))
    config = {}
    pipe = SimpleNamespace(transformer=transformer, scheduler=SimpleNamespace(
        config=config, sigmas=torch.tensor([.8, .4, 0.]),
        register_to_config=lambda **kw: config.update(kw)))
    exp = FluxExperiment(pipe, 'sega_official', (128, 128))
    transformer.eval()
    x = torch.randn(1, 64, 8)
    ids = torch.zeros(64, 3)
    ids[:, 1] = torch.arange(8).repeat_interleave(8)
    ids[:, 2] = torch.arange(8).repeat(8)
    kwargs = {'encoder_hidden_states': torch.randn(1, 3, 12),
              'pooled_projections': torch.randn(1, 6), 'guidance': torch.tensor([4.5]),
              'img_ids': ids, 'txt_ids': torch.zeros(3, 3), 'return_dict': False}
    def run():
        exp.reset()
        latent = x.clone()
        for step in range(2):
            v = transformer(hidden_states=latent, timestep=pipe.scheduler.sigmas[step:step+1],
                            **kwargs)[0]
            latent = latent - .4 * v
        return latent
    baseline = run()
    with SameStateProbe(exp, [0], [direction()]) as probe:
        probed = run()
    assert torch.equal(probed, baseline)
    assert probe.records[0]['replay_max_abs_velocity_error'] == 0
    assert any(abs(v) > 1e-8 for v in probe.records[0]['directions'][0]['jacobian'])
    exp.close()
