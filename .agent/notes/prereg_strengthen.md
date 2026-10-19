# Evidence-strengthening preregistration

Registered 2026-10-19T04:05:00Z, before any generation or scoring for this campaign.
This original text is immutable. Append dated amendments with reasons; preserve
the original and disclose whether relevant outputs existed when amending.

## Purpose and fixed design

Confirm or qualify the finished findings paper at commit fdee66d. No method or
hyperparameter selection on these experiments. Preserve all negative outcomes.
E1 and E2 must be queued before implementing E3. Existing workers are unchanged.

E1 uses 150 previously unused Aesthetic-4K TRAIN captions, seed 0, Krea transformer,
4096x4096, 28 steps, guidance 4.5 and the same scheduler/model/dtype settings as
the completed main tasks. Methods: sega_official, temp_only on sega_official RoPE
with temperature sref, sega_ctm eta 1.0, constancy_tm on sega_official RoPE with
temperature sref and calib/krea_profile_tm.npz (eta 1.0), dype, ntk2. Generation
tasks use prefix 80_, seven-prompt chunks, methods interleaved by chunk. Outputs
are out_krea/fresh/<method>/4096x4096.

Pin the dataset revision and retain metadata hashes. Order TRAIN records by their
stable source file identifier, with original metadata row index as a tie breaker.
Use numpy.random.default_rng(2026).permutation on this ordered list, traverse once,
and take the first 150 eligible unique captions. Normalize captions using Unicode
NFKC, casefold and collapsed whitespace for exclusions. Exclude every caption in
prompts/aesthetic4k_eval.jsonl and all other project prompt manifests or generated
sidecars, including calibration, development, diagnostics and reserved fresh sets.
Record the exclusion sources and their hashes. Exclude unsafe captions (sexual or
graphic violence content) and identifiable real-person requests; conservatively
exclude human portrait/person scenes if identity cannot be resolved from text.
Use a deterministic text filter followed by a caption-only manual audit, before
generation. Record every rejected candidate and reason; replacements come from the
same permutation. Do not inspect generated outputs to select or replace prompts.
Save fresh150.jsonl, a matching reference manifest, and the selection receipt on
work. Download caption metadata and only the selected reference images to scratch;
no full dataset download. Never modify selected caption text.

E2 uses existing held-out manifest rows 8 through 194 inclusive, 187 prompts, seeds
1 and 2, for sega_official, temp_only, constancy_tm and sega_ctm, with the same flags
as E1. Reuse verified matching seed-1 images already present, without regeneration.
Retain unique (method, prompt_id, seed) keys and source paths. Prefix 81_; use chunks
of at most seven, interleaved by chunk/method/seed. Combine with existing seed 0
only after checking settings and prompt identity. Row 7 never enters the analysis.

E3 is a second-family extension on the same fresh150, seed 0: Qwen-Image,
ntk2-equivalent base, released sega_official and temp_only on the SEGA base.
Use SEGA's official Qwen 4K settings and record the exact resolution, steps,
guidance, schedule, positional scaling and temperature in a dated pre-generation
amendment after inspecting the released source. No quality-driven tuning.
CPU tiny-config tests precede a 70_qwen_smoke task; only a successful GPU smoke
allows prefix 82_ campaign tasks. Download Qwen/Qwen-Image once to HF_HOME if absent.
The smoke uses a previously used development caption, not fresh150. Report its
numerical/implementation checks separately from quality. Qwen is an exploratory
transfer assessment, not a second confirmatory test of H1-H3.

## Metrics and exact inferential rules

Primary E1 metrics are ImageReward (IR), PickScore and full-resolution MUSIQ using
the same implementations/model revisions/preprocessing as the existing campaign.
ImageReward uses the mend environment. Delta always means candidate minus released
sega_official except the explicitly reversed H2 contrast. Higher is better for
these three metrics.

H1: temp_only is equivalent to sega_official on both IR and PickScore. Support H1
only if the entire two-sided 95% paired bootstrap CI lies strictly inside
(-0.05, +0.05) for IR AND (-0.10, +0.10) for PickScore. Report each component. A
CI crossing zero alone is not equivalence. Otherwise H1 is not established; call
the outcome incompatible with equivalence only if the CI is wholly beyond a margin.

H2: sega_official beats temp_only on full-resolution MUSIQ. Support H2 only if the
95% CI for sega_official minus temp_only has lower bound strictly greater than 0.
If its upper bound is below 0, report the reverse finding; otherwise unresolved.

H3: neither constancy_tm nor sega_ctm improves IR by more than 0.03 over
sega_official. Support each component if its candidate-minus-SEGA 95% CI has upper
bound <= 0.03; support H3 only if both do. This rules out an improvement larger
than the registered margin; it does not prove identical quality or no improvement
at all. If a lower bound exceeds 0.03, report contradiction; otherwise unresolved.

All CIs: 2,000 paired prompt-bootstrap draws; numpy.random.default_rng(0), recreated
for each method/metric/contrast; resample prompts with replacement, compute the
mean paired difference, take numpy percentile 2.5 and 97.5 (default linear method).
Use manifest prompt order, finite numeric values and float64 arithmetic. Means use
the same percentile bootstrap. Report full precision in CSV/JSON and rounded tables.
These are nominal, unadjusted per-contrast intervals. H1 and H3 are conjunctions;
no familywise error control or post hoc multiplicity correction is claimed.

Secondary metrics: CLIP, crop-1024 MUSIQ, full and crop CLIP-IQA, spectrum log RMSE,
spectrum slope error and tiling1024, matching existing scripts. Spectral/tiling
errors are lower-is-better. Include all measured metrics, not only favorable ones.
FID/KID are descriptive aggregate metrics with the selected references only; at
n=150 they are noisy and cannot establish H1-H3. Retain evaluator settings and
per-image data. Do not compare raw reward scores across model families as causal
family effects. Compare each family internally on matched prompts/noise seeds.

E2 primary analysis averages seeds 0,1,2 within each (method, prompt, metric), then
bootstraps paired prompt-level means using the same 2,000-draw protocol. The unit
is the prompt (n=187), not 561 independent images. Report seed-specific deltas and
the additional-seeds-only (1,2) prompt-mean sensitivity analysis. Apply the same
H1-H3 margins as replication summaries, labeled separately from E1 confirmation.
Do not pool E1 and E2, and do not choose a seed subset based on scores.

E3 reports within-family paired means/deltas with the same bootstrap and fresh-set
ordering, all primary and secondary metrics available for Qwen. H1/H2-style margins
may be shown as exploratory transfer summaries; H3 is not tested because its two
controllers are absent. Runtime/peak-memory evidence is descriptive and includes
settings/offload/backend. No matched-cost or SOTA claim follows.

## Integrity, missingness and stopping

Freeze task/source/config hashes and record actual settings in image sidecars.
Check exact prompt IDs/text, seed, method, dimensions, model/checkpoint, schedule,
steps, guidance and controller settings before evaluation. Verify no duplicate keys,
nonfinite scores, path mismatch or accidental seed alias. Reuse only validated
existing images. Operational retries keep identical settings and preserve failures.
Any implementation correction requires an amendment before regenerated outputs.

Publish final hypothesis decisions only for a complete registered block: all 150
E1 prompts for six methods, all 187 E2 prompts with three seeds for four methods,
or all 150 Qwen prompts for three methods. Intermediate completion counts and
diagnostics may be reported but cannot stop a campaign on statistical grounds.
Missing or invalid samples are reported explicitly, with no silent dropping or
imputation. If compute/time prevents completion, label results incomplete and
provide a clearly separated complete-pair descriptive table; do not claim support
for the registered hypotheses from an incomplete block.

Evaluation tasks use prefix 90_ and are released only when their dependencies are
complete. Preserve raw scores, prompts, task manifests, preregistration and analysis
outputs on work for reproducibility. Keep large images/weights on scratch. Update
STATE.md, results.md and HANDOVER.md after each meaningful step. Revise paper claims
when findings fail the registered hypotheses; distinguish unestablished from
contradicted, observational metric evidence from human preference, and released
code from paper equations. No fresh human study or universal transfer is claimed.

## Timestamp correction, 2026-10-19T04:02:37+00:00

The header timestamp was typed ahead of the actual clock. The initial file existed
at 2026-10-19T04:02:07Z (tool receipt); this amendment records the actual clock.
No new generation or scoring has been queued or run. Original text SHA256:
`06e1974c93fa8b7c344456b7dd2300d597a502f034ac476f5a685d6c6ccbde69`. No design or analysis rule changes.
