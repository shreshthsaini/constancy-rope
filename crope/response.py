"""Same-state finite differences around released SEGA, without changing its trajectory.

This is a diagnostic, not a new sampler. Only the ordinary baseline output reaches
the pipeline scheduler. Extra calls bypass the transformer's outer hooks, while
its projected-feature SEGA hook still runs. All probe effects are then restored.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

import torch

from .spectral import axis_power, token_grid


@dataclass(frozen=True)
class Direction:
    name: str
    vector: torch.Tensor
    preserve_axes: tuple = ()


def band_directions(experiment):
    """Separate spatial-axis octave controls, plus a spatial scalar control."""
    dims = experiment.axes_dims
    directions = []
    for axis in (1, 2):
        freq = experiment.rope.frequencies(axis).double() / (2 * math.pi)
        # Include every represented RoPE pair, even below the grid's DFT limit.
        exponents = torch.floor(torch.log2(freq)).long()
        for exponent in sorted(set(exponents.tolist())):
            vector = torch.zeros(sum(dims), dtype=torch.float64)
            offset = sum(dims[:axis])
            vector[offset:offset + dims[axis]] = (exponents == exponent).repeat_interleave(2)
            # A full-axis mask is exactly canceled by RMS preservation.
            if (exponents == exponent).all():
                continue
            directions.append(Direction(f'axis{axis}_octave{exponent}', vector, (axis,)))
    vector = torch.ones(sum(dims), dtype=torch.float64)
    vector[:dims[0]] = 0
    directions.append(Direction('spatial_scalar', vector))
    return directions


def perturb_gain(base, direction, amount, axes_dims):
    """Preserve gain-vector RMS in FP32, not query/key activation energy."""
    vector = direction.vector.to(device=base.device, dtype=base.dtype)
    if vector.shape != base.shape or not torch.isfinite(vector).all():
        raise ValueError('Direction must be a finite full gain vector')
    if vector[:axes_dims[0]].count_nonzero() or not torch.equal(vector[::2], vector[1::2]):
        raise ValueError('Directions must preserve the index axis and rotary pairs')
    if not math.isfinite(amount) or not torch.isfinite(base).all() or not (base > 0).all():
        raise ValueError('Finite amount and positive finite gains required')
    if any(axis not in (1, 2) for axis in direction.preserve_axes):
        raise ValueError('RMS preservation is only defined for spatial axes')
    if amount == 0:
        return base.clone()
    gain = base * torch.exp(.5 * amount * vector)
    for axis in direction.preserve_axes:
        start, end = sum(axes_dims[:axis]), sum(axes_dims[:axis + 1])
        gain[start:end] *= (base[start:end].square().sum() / gain[start:end].square().sum()).sqrt()
    if not torch.isfinite(gain).all() or not (gain > 0).all():
        raise ValueError('Perturbed gains overflowed or underflowed')
    return gain


def _velocity(output):
    return output[0] if isinstance(output, tuple) else output.sample


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


@contextmanager
def controller_transaction(experiment):
    """Restore exact baseline objects, including history identity, on exceptions."""
    controller = experiment.controller
    qk, logg = controller.qk_scale, controller.log_g
    history, entries = controller.history, list(controller.history)
    diagnostics = experiment.official.diagnostics
    try:
        yield
    finally:
        controller.qk_scale, controller.log_g = qk, logg
        history[:] = entries
        controller.history = history
        experiment.official.diagnostics = diagnostics


class SameStateProbe:
    """Temporarily attach a diagnostic to an eval-mode, non-offloaded FluxExperiment.

    At selected steps: one unperturbed replay, then +/-epsilon and +/-epsilon/2
    per control. Replays receive cloned inputs and the baseline's initial RNG.
    The half-step measurements are reserved for checking local prediction.
    """

    def __init__(self, experiment, steps, directions=None, epsilon=.1, sink=None):
        if experiment.method != 'sega_official' or experiment.official is None:
            raise ValueError('Same-state diagnostic requires released sega_official')
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be positive and finite')
        self.experiment = experiment
        self.transformer = experiment.pipe.transformer
        if self.transformer.training or any(module.training for module in self.transformer.modules()):
            raise ValueError('Probe requires an eval-mode transformer')
        if hasattr(self.transformer, '_hf_hook') or hasattr(self.transformer, '_orig_mod'):
            raise ValueError('Offloaded or compiled transformer is unsupported by this diagnostic')
        self.steps = set(steps)
        if not self.steps or any(not isinstance(step, int) or step < 0 for step in self.steps):
            raise ValueError('Select nonnegative integer step indices')
        self.directions = band_directions(experiment) if directions is None else list(directions)
        if len({d.name for d in self.directions}) != len(self.directions):
            raise ValueError('Direction names must be unique')
        self.epsilon, self.sink = epsilon, sink
        self.records, self.extra_calls = [], 0
        self.active = None
        self.replaying = False
        self.reference_gain = None
        self.realized_gain = None
        self.handle = None

    def __enter__(self):
        if self.handle is not None or 'forward' in self.transformer.__dict__:
            raise ValueError('Refuse nested probes or an existing forward override')
        self.original_forward = self.transformer.forward
        self.handle = self.transformer.x_embedder.register_forward_hook(self._intervene)
        self.transformer.forward = self._forward
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        self.handle = None
        del self.transformer.forward
        self.active = None
        self.replaying = False

    def _intervene(self, module, args, projected):
        controller = self.experiment.controller
        if not self.replaying:
            self.reference_gain = controller.qk_scale.detach().clone()
            return
        # Hold the actual baseline allocation fixed across local interventions;
        # repeated CUDA spectral reductions must not change the operating point.
        if self.active is None:
            controller.qk_scale = self.reference_gain.clone()
        else:
            direction, amount = self.active
            controller.qk_scale = perturb_gain(self.reference_gain, direction, amount,
                                               self.experiment.axes_dims)
        controller.log_g = 2 * controller.qk_scale.log()
        self.realized_gain = {'float32': controller.qk_scale.float().cpu().tolist(),
                              'cast_bfloat16': controller.qk_scale.bfloat16().float().cpu().tolist()}

    def _statistics(self, velocity):
        exp = self.experiment
        clean = exp.current_xt.float() - exp.sigma * token_grid(velocity.detach().float(), exp.grid)
        axes = axis_power(clean)
        values = torch.cat([a.power.clamp_min(1e-30).log() for a in axes]).double().cpu()
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite same-state spectral statistics')
        return values, [{'axis': axis + 1, 'lower': a.lower.cpu().tolist(),
                         'upper': a.upper.cpu().tolist()} for axis, a in enumerate(axes)]

    def _forward(self, *args, **kwargs):
        exp = self.experiment
        if exp.step not in self.steps:
            return self.original_forward(*args, **kwargs)
        if torch.is_grad_enabled():
            raise ValueError('Run the probe under inference_mode or no_grad')
        inputs = _clone((args, kwargs))
        device = exp.current_xt.device
        devices = [device.index] if device.type == 'cuda' else []
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if devices else None
        baseline = self.original_forward(*args, **kwargs)
        base_velocity = _velocity(baseline)
        base_stat, bins = self._statistics(base_velocity)
        if devices:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        record = {'step': exp.step, 'sigma': exp.sigma, 'model_timestep': exp.model_timestep,
                  'statistic': 'log axis power of x_t - sigma * v at identical x_t',
                  'bins': bins, 'baseline': base_stat.tolist(), 'epsilon': self.epsilon,
                  'directions': [], 'extra_calls': 0,
                  'baseline_gain_float32': self.reference_gain.float().cpu().tolist(),
                  'baseline_gain_cast_bfloat16': self.reference_gain.bfloat16().float().cpu().tolist(),
                  'timer_scope': 'Replay forwards and their statistics; excludes initial cloning, '
                                 'baseline statistics, and sink writes. Use full run times for accounting.'}

        def evaluate(direction=None, amount=0):
            with controller_transaction(exp):
                torch.set_rng_state(cpu_rng)
                if devices:
                    torch.cuda.set_rng_state(cuda_rng, device)
                self.active = None if direction is None else (direction, amount)
                self.replaying = True
                try:
                    # __call__ would run the outer sampler hooks again: call the
                    # implementation directly so step and previous_x0 stay fixed.
                    call_args, call_kwargs = _clone(inputs)
                    output = self.original_forward(*call_args, **call_kwargs)
                    self.extra_calls += 1
                    record['extra_calls'] += 1
                    velocity = _velocity(output)
                    stat, _ = self._statistics(velocity)
                    error = float((velocity.float() - base_velocity.float()).abs().max())
                    return stat, error, self.realized_gain
                finally:
                    self.active = None
                    self.replaying = False

        with torch.random.fork_rng(devices=devices):
            _, replay_error, _ = evaluate()
            record['replay_max_abs_velocity_error'] = replay_error
            if replay_error != 0:
                raise RuntimeError('Unperturbed replay differs from baseline; response is not identified')
            for direction in self.directions:
                plus, _, plus_gain = evaluate(direction, self.epsilon)
                minus, _, minus_gain = evaluate(direction, -self.epsilon)
                half_plus, _, half_plus_gain = evaluate(direction, self.epsilon / 2)
                half_minus, _, half_minus_gain = evaluate(direction, -self.epsilon / 2)
                jacobian = (plus - minus) / (2 * self.epsilon)
                record['directions'].append({
                    'name': direction.name, 'vector': direction.vector.tolist(),
                    'preserve_axes': list(direction.preserve_axes),
                    'realized_gains': {'plus': plus_gain, 'minus': minus_gain,
                                      'half_plus': half_plus_gain, 'half_minus': half_minus_gain},
                    'plus': plus.tolist(), 'minus': minus.tolist(),
                    'half_plus': half_plus.tolist(), 'half_minus': half_minus.tolist(),
                    'jacobian': jacobian.tolist(),
                    'half_plus_residual': (half_plus - base_stat - self.epsilon / 2 * jacobian).tolist(),
                    'half_minus_residual': (half_minus - base_stat + self.epsilon / 2 * jacobian).tolist()})
        if devices:
            torch.cuda.synchronize(device)
        record['extra_seconds'] = time.perf_counter() - started
        self.records.append(record)
        if self.sink is not None:
            self.sink(record)
        return baseline
