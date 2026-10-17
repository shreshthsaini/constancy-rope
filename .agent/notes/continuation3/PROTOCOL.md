# Third development diagnostic

Frozen before generation on 2026-10-17. The user has assigned full project ownership
to the current main agent, including experiments, evaluation, and paper maintenance.
The prior session's monitor-only operating restriction was specific to that session.

## Why these experiments

Two completed campaigns failed to improve on released SEGA. Re-reading all three
existing development probes also rejects a universal positive gain-to-image-power
response: one prompt has no positive width diagonal among six tested bands. The
old probe perturbed both spatial axes on NTK2; it did not establish the response
sign under SEGA. Existing released-SEGA static allocation already fails the dev
gate, so repeating that control is unnecessary.

SEGA centers query/key multipliers before squaring and clipping. Thus a centered
allocation also changes mean logit coefficients. For unclipped m=m_ref*(1+a),
mean(a)=0 implies mean(m^2)=m_ref^2*(1+mean(a^2)). This is a coefficient identity,
not a claim about attention entropy or variance. Two controls separate this scalar
effect from allocation. A third aligns the spectral target with the timestamp of
the delayed clean estimate. It addresses specified historical behavior, not a
bug that invalidates the completed negative results.

## Frozen variants

1. sega_rms_uniform: each spatial axis uses sqrt(mean(m_d^2)) from that run's
   released-SEGA projected features. Preserve axis 0 exactly.
2. sega_rms_normalized: scale each spatial m_d vector to RMS m_ref=s^0.08,
   using released SEGA's longest-side scale. Do not clip again. Preserve axis 0.
3. sega_ctm_lag: eta=1, same SEGA composition and current-sigma visibility gate;
   evaluate only the native target at the previous clean estimate's scheduler sigma.
4. sega_official: fresh baseline on the same hardware and source revision.

No parameter sweep, learned controller, new backbone, or training. Each workflow
generates exactly development rows 0-7 at 4096x4096, seed 0, 28 steps, guidance 4.5,
the existing Krea checkpoint, and mu=1.15. Calibration and native profile unchanged.
All four workflows use the same GPU architecture and separate outputs.
The Blackwell submission was rejected before any job existed because all three
qgb submit slots were occupied. The frozen fallback is two GH200 nodes sharing
the same finite spool; no candidate settings or image counts change.

## Evaluation and decisions

Use fresh dev3_sega_official as the prespecified primary reference to match hardware
and source revision. Compare that baseline with the archived pilot as a separate
reproducibility check. Evaluate ImageReward, CLIP, PickScore, full/crop MUSIQ, and
full/crop CLIP-IQA with unchanged scorer revisions and crop settings. Omit expensive
reference-distribution metrics from this development gate. Do not treat dev CIs as
confirmatory findings.

Retain the old eligibility rule: ImageReward delta > 0, full MUSIQ delta > 0,
CLIP delta >= -0.3. Select at most two by ImageReward delta, then MUSIQ delta,
then method name. Report PickScore and crop quality even if they disagree.
No old held-out data is used to choose settings. If a candidate passes, freeze it
and use a separate, previously uninspected confirmation prompt set before any
efficacy claim. Merely rerunning the old 187 prompts is not fresh confirmation.
If none passes, end this diagnostic and preserve its negative result.

## Cost and execution

32 images. Prior same-workload GH200 timing is about 215 seconds/image at high
GPU utilization. With two GPUs, generation is roughly 57 minutes per GPU, plus
loading and scoring. Plan approximately 70 minutes (60-90 minutes) on two nodes,
with a 90-minute walltime ceiling, about 2.3 node-hours centrally and at most 3
node-hours reserved. This packs four workflows into one shared allocation.
Exact service-unit charges will be taken from accounting.

Live balance before preparation: 8212.919 SUs remaining of 40000, used 31787.081
(79.47%), below the 96% stop threshold. Recheck taccinfo immediately before submission.
No prior allocations are cancelled or repurposed. Old taskq and STOP are preserved.
New isolated spool: /scratch/09032/saini_2/constancy_rope/taskq_dev3.
Generation writes per-image manifests and W&B progress; evaluation logs W&B metrics.
The shared GPU logger records every allocated device. Offline W&B is explicit if no key
is configured. Workers exit after their finite tasks, and the recorded logger child
is stopped and reaped after worker completion.

## Prior-art boundary

SEGA already adapts per-frequency gains from evolving spectral content:
https://arxiv.org/html/2605.22668v1 . DyPE already uses a timestep-dependent positional
schedule: https://arxiv.org/abs/2510.20766 . Query/key normalization and temperature
are established: https://aclanthology.org/2020.findings-emnlp.379/ . These new modes
are causal controls and a timing correction, not a claimed novel normalization
principle or entropy theorem. A stronger mechanism would need its own evidence.

## Evidence

- existing_dev_diagnostic.json: development controls, all existing probes, correction activity.
- Controller audit: lag-only native band-deficit mean absolute differences 0.0393
  (height) and 0.0445 (width); unit-gate sensitivities, not measured quality changes.
- Existing dev2 replay shows small late mapped-correction changes under alignment;
  no prediction that timing alone rescues quality is made.
