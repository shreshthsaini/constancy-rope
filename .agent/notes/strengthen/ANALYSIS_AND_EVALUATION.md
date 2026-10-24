# Strengthening analysis and evaluation

Updated 2026-10-19 UTC. The generation queue is owned by the root session.
These helpers use the existing fleet. They do not submit jobs or poll the scheduler.

`scripts/strengthen_analysis.py` has 25 passing tests. The evaluation helper has
38 passing tests, including both diagnostic schedules, publication moves,
method shards, strict merges and synthetic FID/KID equality. The root session
owns the combined campaign test receipt.

## One-shot evaluation

Run these commands from the project root on the current shared CPU workspace:

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export UV_CACHE_DIR=/scratch/09032/saini_2/constancy_rope/cache/uv
crope_py() { uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/crope/bin/python python "$@"; }
```

Each invocation checks the complete generation block. It returns exit status 2
while images or completed task scripts are missing. No evaluation task is staged
or published in that case. Once the block passes config, schedule, PNG hash,
source hash and reference checks, `--publish` releases prefix `90_` tasks:

```bash
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/factorial_queue.json --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --publish
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/ckpt_matrix_queue.json --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --publish
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/tsweep_queue.json --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --publish
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/default_shift_queue.json --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --publish
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/fresh_queue.json --references prompts/fresh150_refs.jsonl --publish
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/seeds_queue.json --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --publish
```

Omit `--publish` to stage scripts after successful validation. A persistent
publication reservation prevents a duplicate release while a worker moves a task
between queue directories. An existing failed task requires inspection. A
reservation with no visible task also requires inspection; the helper does not
reissue it automatically.

Each score directory with more than 200 new images gets one evaluator task per
method. Smaller groups get one evaluator task. ImageReward remains one task per
score directory. Seeds 1 and 2 use separate score directories, after the complete
generation block passes. The evaluator uses the original ten metric groups, four shared 1024
crops, crop seed 0, evaluator bootstrap default 1000, and the existing model
caches. The downstream paired analysis uses 2000 bootstrap draws. ImageReward
runs in the existing `mend` environment. Evaluator `--allow-missing` preserves
its diagnostic reports, but a task's final CPU validation refuses to seal a
missing score, changed setting, incomplete summary or invalid row. Such a task
fails with its reports retained for inspection.

Scores go to `/scratch/09032/saini_2/constancy_rope/eval_krea/strengthen/`:

| Block | Score directory | New images | Evaluator tasks | IR tasks |
| --- | --- | ---: | ---: | ---: |
| Factorial | `factorial` | 374 | 2 | 1 |
| Checkpoint and guidance | `ckpt_matrix` | 32 | 1 | 1 |
| Sweep | `tsweep` | 165 | 1 | 1 |
| Default flow shift | `default_shift` | 8 | 1 | 1 |
| Fresh | `fresh` | 900 | 6 | 1 |
| Seeds | `seeds_seed1`, `seeds_seed2` | 264, 374 | 2, 2 | 1, 1 |
| Qwen | `qwen` | 450 | 3 | 1 |

Each score directory ultimately holds `per_image.csv`, `ir.csv`, `metrics.json`
and `summary.csv`. Method shards first write under `<score directory>/shards/<method>/`
using the corresponding method input subroot. Every evaluator task seals output
hashes in `evaluation_receipt.json` before it can reach the queue's done folder.
Gate status, the immutable evaluation plan, task hashes, validated image records and dependency checksums live
under `.agent/notes/strengthen/evaluation/<block>/`. Task scripts live outside
the image roots. Keep every dependency named in `dependencies.sha256` unchanged
while its evaluation tasks are pending or running.

After all evaluator tasks for a score directory finish, run the one-shot CPU
merge. It checks completed task hashes, all source and input hashes, shard output
hashes, method/prompt/seed/path identities, exact population coverage, metric
settings, scorer provenance and CSV agreement with the evaluator JSON reports.
Missing shards return exit status 2 without creating canonical outputs. Corrupt
or duplicate rows raise an error. Repeating a successful merge preserves
identical canonical bytes and receipts.

```bash
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/factorial_queue.json --merge
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/ckpt_matrix_queue.json --merge
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/tsweep_queue.json --merge
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/default_shift_queue.json --merge
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/fresh_queue.json --merge
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/seeds_queue.json --merge
```

Single evaluator groups are validated by the same command without rewriting
their reports. A completed seed group can merge while the other seed remains
pending. ImageReward runs independently; `--merge` does not claim IR completion
or create IR scores. Wait for its matching `90_ir_` task before paired analysis.

The merged scalar rows follow method, original prompt-manifest order and seed.
Per-method FID/KID and patch FID/KID estimates and confidence endpoints are copied
unchanged from shard reports. `distribution_bootstrap` creates its RNG inside
each call and sorts generated and reference prompt keys, so task order does not
change its Monte Carlo draws. A synthetic test checks exact full-root versus
method-shard estimates and endpoints for FID and KID, with four crops and two
seeds. Scalar summaries likewise restart their RNG per call.

Every shard receives the unchanged complete reference manifest. Spectral
reference profiles average all those reference images, independently of the
generated-image root. PyIQA and Inception crop coordinates depend only on image
size, prompt ID, crop seed and crop count. No cross-method paired distribution
aggregate is inferred from marginal summaries: merged `paired_deltas.csv` is
empty and the report states this explicitly. The registered paired scalar
analysis remains in `strengthen_analysis.py` with its original settings.

The checkpoint matrix validates each guidance value, checkpoint revision, direct
CUDA seed 42, official DyPE provenance and the executed mu 1.15 schedule. The
default-shift diagnostic separately requires seed 0, guidance 4.5, the unchanged
4096 sequence anchor and the executed mu 11.55 schedule. Neither diagnostic enters
the confirmatory hypothesis family. Their scores remain descriptive development
evidence.

## Qwen sequence

The smoke task is separate from the 450-image transfer block. After its passed
receipt exists and the root session verifies it, prepare the full block with:

```bash
crope_py scripts/strengthen_qwen_queue.py qwen
```

That command only stages the 66 generation tasks. The root session publishes them
with the existing queue publisher after checking the receipt. Once those tasks
finish, the same evaluation gate applies:

```bash
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/qwen_queue.json --references prompts/fresh150_refs.jsonl --publish
crope_py scripts/strengthen_evaluation.py .agent/notes/strengthen/qwen_queue.json --merge
```

Generation uses `out_qwen/fresh/<raw arm>/4096x4096`. Raw arm identities are
`qwen_base`, `qwen_sega` and `qwen_temp_only`. Analysis maps them to `base_default`,
`sega_official` and `temp_only_sbase`. The Qwen base uses its default extrapolation
with centered coordinates and theta 10000; it is not an NTK-scaled control.

The E3 manifest checks the frozen `fixed_config` and source/version provenance.
The sidecar audit independently checks the original caption, effective suffix,
CUDA seed reset to 0, 40 finite steps, and the realized 41 sigmas and 40 timesteps
for mu 0.9 and terminal shift 0.02. The two-step GPU smoke supplies execution
evidence only.

## Follow-up analysis

After the paired evaluator and ImageReward outputs exist, run the matching
manifest. The original sources are combined by resolved PNG path and explicit
method aliases. Historical seed-1 images remain at their original paths.

```bash
crope_py scripts/strengthen_analysis.py .agent/notes/strengthen/analysis_P1.json .agent/notes/strengthen/analysis/P1
crope_py scripts/strengthen_analysis.py .agent/notes/strengthen/analysis_P3.json .agent/notes/strengthen/analysis/P3
crope_py scripts/strengthen_analysis.py .agent/notes/strengthen/analysis_E1.json .agent/notes/strengthen/analysis/E1
crope_py scripts/strengthen_analysis.py .agent/notes/strengthen/analysis_E2.json .agent/notes/strengthen/analysis/E2
crope_py scripts/strengthen_analysis.py .agent/notes/strengthen/analysis_E3.json .agent/notes/strengthen/analysis/E3
```

An incomplete block writes a missingness report and descriptive complete-pair
tables, then returns exit status 2. It cannot support a registered hypothesis.
Each report archives its inputs and analysis source. An archived report can be
recomputed without scratch files:

```bash
crope_py scripts/strengthen_analysis.py --replay .agent/notes/strengthen/analysis/E1 .agent/notes/strengthen/analysis/E1_replay
```

`means.csv`, `paired.csv`, `prompt_means.csv` and `contrasts_holm.csv` provide the
tables. `coverage.json` records missing or invalid samples. `hypotheses.json`
labels fresh-only, replication and Qwen summaries as descriptive or exploratory.
The amended confirmatory decisions are in E1's `confirmation_family.json` and
`confirmation_family.csv`: pooled historical 187 plus fresh 150 for H1, fresh 150
only for H2, with Holm across the three elementary tests. H3 remains historical.

P1 reports eight factorial contrasts across ten metrics, a fixed Holm family of
80. P3 reports four contrasts against the frozen sref control across ten metrics,
a family of 40. Qwen has two contrasts against released SEGA across ten metrics,
a family of 20. E2 averages seeds within each prompt before resampling. Its sample
size is 187, with separate seed-specific and seeds-1-and-2 summaries.

## Review corrections and current evidence

The root review caught a publication-path error before any evaluation was queued.
Generation publication renames scripts out of staging. The gate now reads and
hashes `taskq/done/<task name>`, after checking completion. Tests reproduce the
staging-to-pending-to-running-to-done sequence and pass with staging absent.

The original paper bootstrapped the archived candidate CSV row order. The
retrospective Holm audit preserves that order so it cannot move the published
nominal endpoints. New analyses follow manifest order. Historical means and
intervals match the original 60-row paired table within 5e-16. The full family
retains the partial, nonrandom Constancy sample of 41 prompts.

The retrospective audit is saved at
`.agent/notes/strengthen/historical_holm/historical_holm.csv`. Of the nominally
significant intervals, 28 of 39 survive Holm in the full 60-cell table and 16 of
21 survive in the condensed 30-cell table. The allocation MUSIQ effect survives
both. The condensed table loses marks for base-plus-temperature CLIP-IQA, DyPE
CLIP and MUSIQ, Constancy-TM CLIP, and SEGA-plus-TM CLIP. This is a retrospective
multiplicity audit, not new confirmation.

Real-source preflights accepted all 561 existing P1 baseline rows and all 484
available E2 rows, including 110 reused seed-1 images. They returned incomplete
status because the new images and scores were absent. The analysis agent queued
no evaluation tasks and changed no frozen generation source.
