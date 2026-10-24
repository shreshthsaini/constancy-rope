# Qwen transfer protocol and implementation, 2026-10-19 UTC

All three controls use the released SEGA Qwen transformer and pipeline at clean
commit `ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e`. We do not change those sources.
The model is Qwen/Qwen-Image revision
`75e0b4be04f60ec59a75f475837eced720f823b6`. Download metadata and size evidence are
in `qwen_hub_manifest.json`; the parent agent owns the single serial download.
No Qwen generation occurred while these settings were derived.

## Fixed settings

The released `qwen_sega/run_qwen.py` uses 4096x4096, 40 denoising steps, BF16,
true classifier-free guidance 4.0, empty negative prompt and a CUDA generator reset
to seed 0 for every image. Append its exact suffix
`, Ultra HD, 4K, cinematic composition.` to every caption for generation. Preserve
the original caption as `config.prompt` for evaluator identity and store the
suffixed string separately as `config.effective_prompt`. Maximum conditioning
length is 512 after the released tokenizer/system-template processing.

The released pipeline passes `image_seq_len` as the upper sequence anchor for
its flow shift. The pinned Qwen scheduler has `max_shift=0.9`, so its 4K mu is
0.9, with the original `shift_terminal=0.02`, dynamic exponential shifting and
no user-provided `target_resolution`. Use tiled VAE decoding. On a GH200 with
more than 60 GiB, the released entry point places the whole pipeline on GPU;
we preserve that policy initially and require the same policy in smoke and full
generation. A memory-policy change requires an explicit recorded amendment and
a new smoke; it is not silently applied after an OOM.

## Three arms

| Evaluator identity | Positional base | Spatial Q/K amplitude | Allocation |
|---|---|---|---|
| `qwen_base` | Qwen default theta 10000, centered coordinates | 1 | None |
| `qwen_sega` | Released SEGA | 4^0.08 before allocation | Released spectral allocation |
| `qwen_temp_only` | Released SEGA | Uniform 4^0.08 | None |

All three use the same pipeline, sampler, prompt suffix, CFG and weights. The
baseline concerns Qwen's default positional extrapolation, not the entire default
diffusers sampling package. Its full image and text rotary tensors are exactly
equal to the installed diffusers 0.39.0 default Qwen implementation at the
256x256 packed image grid. A tiny one-layer transformer with identical weights
also gives exactly the same forward output as the default Qwen transformer.
The retained spectral computations in the baseline are discarded and do not
affect its positional tensors. Do not use this implementation for a claim about
native Qwen's minimal runtime.

The SEGA spatial base multiplier is
`4**(2*56/(56-2))/(1+0.1*log(4))`; the executed code has the doubled exponent
despite an outdated nearby comment claiming it was removed. Do not pass
`target_resolution=4096`: that alternate branch changes the denominator to
`1+log(4)` and would not match the released entry point.

Released SEGA uses alpha 0.15, tanh beta 1.5, minimum amplitude 1, spread range
[0,1] and concentration exponent 1.5. Its base Q/K amplitude is `4**0.08`,
equivalent to a spatial logit multiplier `4**0.16=1.2483305489016119` before
allocation. It assigns the width allocation to the height axis through the
literal `mscale_h=mscale_w`; we preserve this behavior as part of the release.
Setting `mscale_alpha=0` gives the uniform temperature control while preserving
the base and all frame-axis behavior. Frame amplitudes remain 1 in both Qwen
SEGA arms. This differs from the FLUX temp_only controller, which applies its
uniform gain to the index axis too. The Qwen comparison removes allocation while
holding that axis fixed.

## CPU checks and GPU gate

`scripts/strengthen_qwen.py check` imports both official modules without model
weights and checks:

- Exact native rotary equality at the full 4K token grid, including text tokens.
- Exact tiny full-transformer native-control equality with identical weights.
- Uniform spatial amplitudes independent of spectrum and spread, frame gain 1.
- Released allocation changes the high-resolution rotary tensors.
- Returning to the base arm after other arms clears the positional cache exactly.

`tests/test_strengthen_qwen.py` also checks arm restoration after interruption,
prompt/seed/scheduler contracts and rejection of corrupt/partial existing outputs.
The final CPU receipt must match current source hashes before staging.

GPU smoke is prefix `70_qwen_smoke`, development row 0, all three arms, two steps
at 4096x4096 with the production CFG, scheduler, generator and VAE settings.
Check finite latents after both steps, exactly two executed steps, successful
4096x4096 PNG decode and sidecar hashes. Two steps exercise repeated forward calls
and CFG state. This smoke is execution evidence only, with no quality selection.
Only successful matching `scratch/diag/qwen_smoke/passed.json` permits prefix
82_ full generation: the frozen fresh150, 40 steps, three arms, 450 images.
Generation uses seven-prompt chunks and preserves identical seed-0 CUDA noise
across prompts as in the released entry point, matched across arms.

`scripts/strengthen_qwen_queue.py qwen_smoke` stages a smoke task after the parent
has published fresh and seeds. Publication should wait for
`qwen_download_receipt.json`. The same script with argument `qwen` refuses to stage
campaign work until the GPU receipt matches source, model revision and placement.
Its manifests are compatible with the parent queue publisher. Evaluate the fresh
root `scratch/out_qwen/fresh`, reference method `qwen_sega`; do not include smoke
outputs in quality results. Qwen is exploratory transfer evidence.
