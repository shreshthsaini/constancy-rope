You are strengthening the evidence of a finished findings paper. Work in /work/09032/saini_2/vista/Projects/Exploration/constancy_rope.
READ FIRST: HANDOVER.md, .agent/STATE.md, paper/PLAN.md, .agent/notes/codex_brief_5.md (previous plan, now superseded by
this file), .agent/notes/prereg_strengthen.md (written by the previous run at 23:02; no generation has happened yet),
.agent/notes/fable_review_2026-10-18.md (senior review; its action list drives this plan), .agent/notes/main2/REPORT.md,
paper/tables/make_tables.py. A GPU fleet (job CR22, 8 GH200 nodes, 24 h, may still be queued) drains
/scratch/09032/saini_2/constancy_rope/taskq/pending via slurm/worker_loop.sh (do not modify it or node_launch.sh).

STEP 0 (before any task is queued): append a dated AMENDMENT to prereg_strengthen.md (never rewrite the original text)
incorporating the changes below, with the reason "senior review before any data were generated". Fix H1 so it is decidable:
use an equivalence margin derived from a meaningful effect (e.g. half of SEGA's held-out IR gain over NTK) and/or pool
fresh150 with the 187 held-out rows for the equivalence test; compute the expected CI width from existing variance and
state it. Add Holm correction across the pre-registered family.

PRIORITIZED CAMPAIGN (queue in this order of filename prefixes so the fleet runs the highest value first):
P1 (prefix 75_) Factorial base x temperature on held-out prompts 8-194, seed 0, same settings as existing main runs:
   arm A = SEGA released RoPE base with NO temperature and no allocation (CROPE_ROPE=sega_official, method temp_only,
   --temperature none, or the cleanest equivalent; verify in code what "none" does); arm B = ntk2 base with SEGA's
   reference temperature (method temp_only, --temperature sref, no CROPE_ROPE override). With existing arms
   (ntk2 = base ntk2/no temp; temp_only_sbase = SEGA base + sref temp; sega_official = + allocation) this completes a
   2x2 factorial plus allocation. Outputs out_krea/factorial/<arm>/4096x4096 with unique method identities for the evaluator.
P2 (prefix 76_) Matched-guidance checkpoint control: OFFICIAL DyPE code (scratch diag/official_dype.py pattern) on dev
   prompts 0-7 x {FLUX.1-dev, FLUX.1-Krea-dev} x {guidance 3.5, 4.5}, seed 42 as before; save under scratch diag/ckpt_matrix/.
P3 (prefix 77_) Temperature sweep on held-out prompts 8-62 (55 rows): SEGA base, logit temperature multipliers spanning
   below and above sref (e.g. 1.0, 1.1, 1.25 (=sref at s=4), 1.33 (logN), 1.45), method temp_only with a temperature
   override you add cleanly (record it in config). Outputs out_krea/tsweep/.
P4 (prefix 80_) E1 fresh150 confirmation as in brief 5 (Aesthetic-4K TRAIN captions, numpy seed 2026, exclude eval
   captions, filter unsafe/real-person; prompts/fresh150.jsonl + refs). Methods: sega_official, temp_only_sbase, ntk2,
   dype, PLUS factorial arms A and B. Drop constancy_tm and sega_ctm from fresh150 (already negative with tight CIs).
P5 (prefix 81_) Seeds 1 and 2 on held-out 8-194 for sega_official and temp_only_sbase only (drop constancy_tm/sega_ctm).
P6 (prefix 70_qwen_smoke then 82_) Qwen-Image second family as in brief 5, arms: base-only (its default extrapolation),
   SEGA released (official qwen_sega), SEGA base + temperature without allocation. Only after P1-P5 are queued.
P7 (no GPU or tiny) MUSIQ input audit: determine exactly what pyiqa MUSIQ does with a 4096^2 input in our eval (resize?
   multi-scale?), document it; and generate 8 dev images with the diffusers DEFAULT flow shift (mu ~11.5) for one method to
   show the schedule collapse (prefix 78_).
Evaluation: queue eval + ImageReward tasks per block as they finish (prefix 90_; patterns from taskq/done/60_eval_mainsb.sh,
61_ir_main.sh). Analysis script(s) for factorial, sweep, fresh, multi-seed, Qwen with paired bootstrap and Holm marks.

PAPER UPDATE (after results): follow the Fable review's claims-audit and figure fixes (sharpening wording vs spectrum,
abstract "diagnostics show why" overclaim, guidance confound, MUSIQ pipeline statement, Holm marks, Figure 1b redesign to
stacked absolute paired gains with CIs, Table 3 to appendix, A.2 heatmap scale, hero selection rule in caption, "tie"
definition at first use). Add the factorial result as the core of the decomposition section; add fresh-set and seed
confirmation; Qwen as a section or appendix. Report results whatever they are; adjust claims if hypotheses fail.
Rebuild with ~/.local/bin/tectonic, render pages with paper/render_pages.sh and inspect, refresh paper/dist PDF + zip.

AFTER EVERY STEP rewrite .agent/STATE.md status (under 120 lines) and append to .agent/notes/results.md; keep HANDOVER.md
current so a fresh agent can resume.

HARD CONSTRAINTS: shared CPU node, OMP_NUM_THREADS=8, own processes under 20, no GPU here. NEVER ssh, sbatch, scancel,
pkill, killall. Never delete anything under /scratch (move aside). No em dashes or section symbol in prose. You cannot
commit; leave changes and list them. If the fleet is not running, poll every 10 minutes with sleep. Final report: what was
queued, results tables, hypothesis outcomes, paper changes, files changed.
