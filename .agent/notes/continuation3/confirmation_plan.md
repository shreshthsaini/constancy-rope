# Frozen fresh confirmation plan

Freeze date: October 17, 2026, before dev3 outcome-based selection. No generation
was queued. Run this confirmation only if a prespecified dev3 candidate passes
the existing development rule. Keep the selected controller and hyperparameters
fixed throughout confirmation.

## Source and selection

Use the 64 prompts in `confirmation64_prompts.jsonl`, from the locally cached
`zhang0jhon/Aesthetic-4K` revision
`8c5d5cb8b94230ff897d87bf060451257d1c7bf8`, source split `eval/size_2048`.
The source contains 2,781 captions. All 195 previous project evaluation captions
also occur in this source and were excluded. The exclusion set includes all
39 Pick-a-Pic calibration captions, including the 16 used for native calibration.
The earlier mechanism probes use the same excluded evaluation prompts.

Normalize captions with Unicode NFKC, casefold, then join regex `\w+` tokens
with spaces. Exclude matches to either prior manifest, previous image stems,
and duplicate normalized captions. This leaves 2,586 eligible rows. Select the
first 64 after sorting the SHA256 of UTF-8 bytes for
`constancy_rope_confirmation_v1 + NUL + source_file_name + NUL + normalized_caption`.
Break a hash tie by source row. This rule uses no generation, score, manual
quality judgment, or semantic category selection. It samples the source caption
distribution without claiming balanced coverage of every subject or style.

All selected source reference files are cached. None was decoded or scored for
selection. IDs use `a4k_confirm2048_<source-image-stem>` and are recorded with
source row, revision, reference path, and selection hash. The complete receipt
and exclusions are in `confirmation64_freeze.json`.

Manifest SHA256:
`965992616563a274ea275d8ef0a7083415800adb5c88d494a37fb272a996e6f2`.

These are unseen-by-project confirmation prompts after the recorded exclusions.
No model-training disjointness claim is made. Normalized exact matching does not
exclude every semantic near duplicate. The source dataset itself was curated;
the project adds no image-quality selection.

## Generation and reporting

Generate at 4096 by 4096, despite the captions' 2048-source provenance. Use the
same frozen Krea transformer, SEGA RoPE, 28 steps, guidance 4.5, scheduler sigmas,
and seed 0. Generate fresh released-SEGA controls on the same hardware and
software as each selected candidate. Use identical prompt-derived seeds per
pair. Record the checkpoint revision, source hashes, actual sigma trajectory,
and output hashes. Preserve every generated pair and every outcome.

Report paired ImageReward and full-image MUSIQ as the two primary measurements.
Also report PickScore, crop MUSIQ, and CLIP, with paired uncertainty over the
64 prompts. Keep the complete table even if the primary measurements disagree.
Any bootstrap must use one fixed seed and identical resampled prompt indices
across methods. If two candidates pass development, label their confirmation
comparisons and account for multiple comparisons before an efficacy claim.

Do not compare 4096 outputs against these 2048 reference images with FID or
spectral metrics and describe that as a matched 4K reference evaluation. The
confirmation's primary measurements need captions and generated images only.
Keep this source split separate from the repeatedly inspected 195-prompt
Aesthetic-4K size4096 table. Confirmation outcomes must not select further
controller variants or hyperparameters.

## Bounded cost and scheduling

One selected candidate plus released SEGA requires 128 images. At the observed
roughly 100 seconds per 4K image on a gb GPU, generation takes about 3.6 GPU-hours.
Two candidates plus SEGA require 192 images, about 5.3 GPU-hours. Allow roughly
5 to 8 GPU-hours total for generation, learned scoring, loading, and a modest
rerun reserve. This is a planning estimate, not a measured SU charge.

Pack the independent prompt chunks into one existing four-GPU allocation when
available, with a shared spool. On four similarly performing GPUs the estimate
is roughly 1.25 to 2 hours after startup if all are available. Reuse baseline
images across selected candidates. Read the live allocation rate and SU balance
before any new submission. If no candidate passes development, run no
confirmation generation and preserve this frozen manifest for provenance.
