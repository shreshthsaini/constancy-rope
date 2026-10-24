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

## AMENDMENT 2026-10-19T04:11:32+00:00: senior-review campaign

Reason: senior review before any data were generated.
The original text and timestamp correction above are retained verbatim. Before this
amendment, taskq/pending and taskq/running were empty and CR22 job1080046 was
PENDING (observed 2026-10-19T04:08:28Z). No new campaign generation or scoring
has occurred. Prior file SHA256: `7c9ef59de8cc17a4236d6259aa297b4b1222051c5d9ce23c32f234f4cfab7bf8`.
This amendment supersedes the original experimental scope and inferential rules
where they differ. Historical outputs used to plan precision are already observed.

### Scope and priority

P1, prefix75_: held-out rows8:194 inclusive, seed0, two new factorial arms:
`sega_base_none` = temp_only, CROPE_ROPE=sega_official, temperature=none;
`ntk2_sref` = temp_only, no CROPE_ROPE override, temperature=sref.
Existing ntk2, temp_only_sbase, and sega_official complete the four base-by-
temperature cells and the allocation comparison. Code inspection confirms none
returns logit multiplier1 on all axes and temp_only never applies a per-band
correction. No OfficialSEGA feature hook is installed for temp_only. The sref
multiplier is exactly4**0.16=1.2483305489016119, not rounded1.25. These controls
use the existing temp_only convention, including its index-axis temperature;
the released-SEGA minus temp_only contrast also includes the index-axis and
clamped mean-gain differences. It cannot isolate allocation shape alone.
Outputs: out_krea/factorial/<arm>/4096x4096; config.method is the unique evaluator
identity, with config.controller_method recording the generation controller.
All Krea runs use4096square,28steps,guidance4.5, batch1, BF16, the pinned Krea
transformer and existing CPU prompt-derived seeds, SDPA policy and mu1.15.

P2, prefix76_: official DyPE (unmodified local flux_dype implementation) on
previously used dev rows0:7 x checkpoints FLUX.1-dev and FLUX.1-Krea-dev x
guidance3.5 and4.5, seed42 on the CUDA generator as in diag/official_dype.py.
Use28steps,4096square, dype_scale2,dype_exponent2. Outputs diag/ckpt_matrix/;
record exact source/model hashes, guidance, scheduler trajectory and image hashes.
This is a descriptive checkpoint-by-guidance diagnostic, not a new held-out test.

P3, prefix77_: held-out rows8:62 inclusive, seed0, SEGA base, temp_only,
absolute logit multipliers1.0,1.1,4**0.16,4/3,1.45. Record exact multiplier in config
for explicit overrides. Arm identities tau1p0,tau1p1,tau_sref,tau_logn,tau1p45.
Reuse the P1 no-temperature images and existing sref images at the two identical
settings with explicit source mapping; generate only the other165 images.
Outputs out_krea/tsweep/. This is a fixed sensitivity curve, with no choice of a
winning temperature or subsequent tuning. Report all five settings.

P4, prefix80_: fresh150 sampling, exclusion, caption-only audit and reference
protocol remain as originally registered. Six arms now are sega_official,
temp_only_sbase,ntk2,dype,sega_base_none,ntk2_sref. Drop constancy_tm and sega_ctm.
P5, prefix81_: held-out187 rows, seeds1/2 for sega_official and temp_only_sbase
only; reuse validated existing seed1 samples (55 per arm) rather than regenerate.
Output layout out_krea/seeds/seed<k>/<method>/4096x4096 keeps evaluator identities
unique; report prompt-level averages of seeds0/1/2 and seeds1/2 sensitivity.
No renewed tests of the two already negative controllers are planned.
P6: begin Qwen implementation only after P1-P5 have been published. Prefix
70_qwen_smoke precedes82_ campaign tasks. Three arms are its default base-only
extrapolation, official qwen_sega, and the released SEGA base/temperature without
allocation. Exact official settings require a further pre-generation amendment
and CPU checks; publish82_ only after successful smoke. Fresh150, seed0; exploratory.
P7: audit installed MUSIQ input pipeline and version from source. Prefix78_ runs
8 dev rows0:7,seed0,sega_official, Krea, guidance4.5,28steps with the diffusers
DEFAULT flow shift (cached max_image_seq_len4096; mu11.55 at65536tokens), compared
with the existing matching official-shift pilot. Save diag/default_shift and the
executed sigmas. Queue after P1-P5, but lexical prefix78_ gives it early execution.
Evaluation tasks use90_ and are published only after each block's artifacts pass
integrity checks. Keep workers and node launcher unchanged. Queue in P1-P5 order,
interleaving methods in seven-prompt chunks, then P6/P7 as above.

### Revised H1, H2, H3 and precision

H1 primary equivalence population pools the187 old held-out seed0 prompts and150
fresh seed0 prompts, n337 distinct prompts. This is explicitly a pooled extension
using already observed old data, not a wholly independent confirmation. The
fresh150-only estimate and CI are always reported separately. No extra seeds
are treated as additional independent prompts. Require both IR and PickScore
equivalence for temp_only_sbase minus released SEGA.
IR margin =0.08197713636363636, half the existing held-out SEGA-over-DyPE IR gain
0.16395427272727273. This meaningful baseline gap is the senior review's suggested
reference; the margin is13.79percent of SEGA's0.5944409465240642 gain over NTK.
Half the NTK gain (0.29722) would allow a much larger difference and is not used.
PickScore margin remains0.10. Freeze these margins before any fresh score.
H1 requires each nominal95percent paired bootstrap CI strictly inside its margin
AND both corresponding Holm-adjusted equivalence p-values <=0.05.
Report unestablished if either condition fails, and incompatible with the margin
only if the nominal CI lies wholly beyond one margin. A zero-crossing CI is not
equivalence. Fresh-only support is not substituted for failed pooled support.

Planning data: existing paired temp-minus-SEGA means IR0.026047796791443843 and
Pick0.028042910570767063; sample SDs (ddof1)0.29033194757462466 and0.4155499092226201.
Normal-approximate full95percent CI widths2*1.959964*SD/sqrt(n):
IR n150=0.09292387,n337=0.06199519; Pick n150=0.13300123,n337=0.08873324.
Conservative family3 Bonferroni-equivalent98.333percent full widths are
IR0.11350100/0.07572344 and Pick0.16245311/0.10838239 at n150/n337.
These are planning estimates assuming new paired variance matches old variance,
not guarantees or actual bootstrap/Holm intervals. At the old point estimate,
the pooled precision is compatible with both fixed margins, including this
conservative benchmark. Existing CSV: .agent/notes/main2/heldout_per_prompt.csv.

H2 remains fresh150-only superiority of sega_official over temp_only_sbase on
whole-image-input MUSIQ: nominal95percent paired CI lower>0 AND Holm-adjusted
one-sided p<=0.05. Reverse or unresolved outcomes are reported explicitly.
H3 is withdrawn from the new confirmation campaign because its controllers are
excluded by the senior-review priority decision. Preserve historical negative
results and the original H3 text; do not call H3 newly confirmed or failed.

### Bootstrap, multiplicity and exploratory contrasts

Retain2000 paired prompt resamples, numpy.default_rng(0) reset per contrast,
float64, manifest order, percentile2.5/97.5 with linear interpolation. For pooled
H1, concatenate old187 in manifest order then fresh150, and resample337 prompts.
Also report separate-set estimates to expose population heterogeneity.
For every paired vector x, let d=mean(x), e_b=mean(resample_b(x))-d. Against boundary c:
p_greater=(1+count(e_b>=d-c))/2001;
p_less=(1+count(e_b<=d-c))/2001.
Equivalence p=max(p_greater at c=-margin,p_less at c=+margin).
Superiority H2 uses p_greater at c=0. These centered, unstudentized bootstrap
p-values are approximate, with plus-one Monte Carlo correction, not exact tests.
Apply Holm stepdown across exactly3 primary component p-values: H1_IR,H1_Pick,H2.
Sort increasing, compare p_(i) to0.05/(3-i+1), stop at first nonrejection;
adjusted p_(i)=min(1,max_(j<=i)((3-j+1)*p_(j))). H1 needs both components.
CIs stay nominal and are labeled as such; Holm marks refer to adjusted p-values.
No primary conclusion until the complete revised fresh block exists and the
pooled archived counterparts pass integrity checks. Missing components use p1
in summaries and can never establish a registered hypothesis.

Exploratory paired factorial contrasts (8): base effect at no temp and at sref;
temperature effect at each base; average base effect; average temperature effect;
interaction=(SEGA-base sref minus SEGA-base none) minus (NTK sref minus NTK none);
released SEGA minus SEGA-base sref. Report10 existing per-image metrics and Holm
across80 two-sided contrasts, two-sided p=min(1,2*min(p_greater,p_less)). Sweep:
4 settings minus sref x10metrics, family40, plus descriptive fixed curve only.
Fresh secondary comparisons:5 arms minus SEGA x10metrics, family50, distinct
from primary family3. Multi-seed: temp-minus-SEGA x10metrics for primary prompt
averages over0/1/2, family10; seeds1/2 and each seed are separately labeled
sensitivities. Qwen:2 arms minus SEGA x10metrics, family20. Existing full and
condensed paper delta tables receive separate retrospective Holm families60
and30; adjusted marks do not convert these old tests into preregistered tests.
No claim of multiplicity control across all exploratory tables is made.
All missingness, immutable-source rules, no quality-based stopping, surrogate-
metric limitations, and complete-block reporting requirements remain in force.

## AMENDMENT 2026-10-19T04:13:38+00:00: source and Monte Carlo clarification

Reason: senior review before any data were generated. P1 has been queued but no
new task is running, done or failed; no generation or scores exist. P2 is not queued.
Direct inspection of the pinned official DyPE yarn branch corrects the P2 shorthand
above: use method=yarn, kappa=t**2 with coefficient1, beta0/beta1=1.25/0.75 and
gamma0/gamma1=16/2. It has no dype_scale/dype_exponent constructor arguments; the
separate unused ntk branch has exponent2*t**2. Preserve the exact prior diagnostic
yarn implementation. Wrapper config will record these fixed source constants.

For exploratory and retrospective TWO-SIDED tests, use the absolute centered
bootstrap tail p=(1+count(abs(e_b)>=abs(d)))/2001. This replaces the doubled
one-sided-tail formula above. The doubled formula has minimum2/2001, too coarse
for the first Holm80 or Holm60 threshold; absolute-tail minimum1/2001 supports
these family sizes. All primary one-sided/TOST formulas, margins,2000 resamples
and nominal CIs remain unchanged. Condensed historical family30 is the5 displayed
methods x6 displayed metrics, exactly the existing main delta table. Full family60
is6 methods x10metrics, including partial Constancy. These remain retrospective.

## AMENDMENT 2026-10-19T04:20:21+00:00: Qwen settings and historical ordering

Reason: senior review before any data were generated. P1-P5 and P7 are queued;
no generation task has run. Qwen weights are downloading, and no Qwen task is
queued. Local official SEGA commitddaab08552c7b7ceba2ebd8f579a0f65dee90b5e is
unmodified. Pin Qwen/Qwen-Image revision75e0b4be04f60ec59a75f475837eced720f823b6.
Qwen outputs use4096square,40steps,BF16,batch1,true_cfg_scale4.0,negative_prompt="",
max_sequence_length512, seed0 with CUDA generator reset per prompt as released.
Append the exact released suffix ", Ultra HD, 4K, cinematic composition." to all
three arms; retain the original caption in config.prompt and record the effective
text separately. Evaluate against original captions. Use official qwen_sega
pipeline/scheduler in all arms, target_resolution=None and VAE tiling. The upper
shift anchor is image_tokens65536, giving mu=max_shift0.9; retain cached terminal
stretch shift_terminal0.02 and all other scheduler settings. No FLUX schedule
assumption transfers to Qwen. Load full pipeline on GPU as released for >=60GB;
if OOM occurs, preserve failure and append a memory-policy amendment before retry.

Arms: qwen_base uses the default Qwen rotary extrapolation, theta10000,
scale_rope=True with centred coordinates and no NTK factor or amplitude gain.
qwen_sega uses official base factor s**(2*d/(d-2))/(1+0.1*ln(s)),s4,d56;
spatial Q/K reference amplitude s**0.08, and released dynamic allocation
alpha0.15,beta1.5,minimum1,spread range0..1,spread exponent1.5. qwen_temp_only
uses the identical released base/reference temperature and alpha0 to disable
allocation. Frame-axis Q/K amplitude remains1 in all Qwen arms. This preserves
the released convention and differs from the FLUX temp_only control's index-axis
gain; make that distinction explicit when discussing transfer.
CPU checks must establish native rotary equivalence at the full256x256 token
grid, tiny identical-weight transformer equivalence, exact temperature amplitudes
independent of spread, and changed rotary output with allocation enabled.
The prefix70_qwen_smoke task uses previously seen dev row0,all3arms, TWO steps
at full4096square, otherwise the same settings. Require finite states at both
steps, a finite decoded image and matching sidecar/image dimensions and hashes.
The smoke is an execution/memory check only. It cannot establish quality,40-step
numerical stability or transfer. Release prefix82_450images only after the smoke
receipt passes and matches the frozen worker/source/model configuration. Seed0
and all150 fresh captions are fixed. Preserve/report every outcome, no tuning.

Historical Holm audit retains the original archived CSV order (lexical prompt IDs)
for its finite2000draw bootstrap, reproducing published nominal intervals. The
original table generator used that order. New analyses, including pooled H1, keep
the manifest order specified above. This explicit historical-order exception
prevents accidental changes to old borderline intervals; it is not selected from
new results. Retrospective families remain60 and30 as already specified.
