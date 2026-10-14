"""Inspect the requested scheduler without loading model weights."""

import math

import numpy as np
import torch


def diagnose_schedule(config, resolution, steps=28):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    if steps <= 0:
        raise ValueError("steps must be positive")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    tokens = math.prod(n // 16 for n in resolution)
    mu = calculate_shift(tokens, config.get("base_image_seq_len", 256),
                         config.get("max_image_seq_len", 4096), config.get("base_shift", 0.5),
                         config.get("max_shift", 1.15))
    sigmas = np.linspace(1, 1 / steps, steps)
    scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
    # Exactly reproduce FluxPipeline's timestep cast and division order.
    model_timesteps = (scheduler.timesteps.to(torch.bfloat16) / 1000).float()
    sigmas = scheduler.sigmas.float()
    return {"res": f"{resolution[0]}x{resolution[1]}", "image_tokens": tokens, "steps": steps,
            "mu": float(mu), "sigmas": sigmas.tolist(), "model_timesteps_bf16": model_timesteps.tolist(),
            "distinct_model_timesteps": int(model_timesteps.unique().numel()),
            "last_evaluated_sigma": float(sigmas[-2]),
            "evaluations_below_sigma_0_3": int((sigmas[:-1] < 0.3).sum()),
            "largest_euler_step": float((sigmas[:-1] - sigmas[1:]).abs().max()),
            "scheduler_config": dict(config), "mode": "unchanged diffusers default"}
