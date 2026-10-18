# Paper rewrite state (lead-writer agent, 2026-10-18)
Goal: rewrite paper/ into findings paper per paper/PLAN.md. Figures owned by figure agent (paper/figs/, do not edit).
Plan: paper/tables/make_tables.py (main/appendix tables + num_results.tex macros), compute_table.py and
mechanism_numbers.py (subagents), new sec/*.tex, old ones moved to sec/archive/. Build with tectonic, render, inspect.
Data: held-out per-prompt archived .agent/notes/main2/heldout_per_prompt.csv (7 methods; constancy n=41).
Seed1: scratch eval_krea/seed1 (+ir/seed1.csv), rows 8-62 n=55. Pilot dev: eval_krea/pilot/4096x4096 (9 methods, n=8).
Status: started. Next: make_tables.py.
## Progress 11:45
- Done: tables/make_tables.py (reproduces registered paired CSV to 5e-16; writes tab_*.tex + num_results.tex, snapshots scratch rows to tables/data/).
- Done: style.tex, macros.tex (names, \dcellB/W/T cells, \N{} accessors), main.tex, sec/intro.tex, sec/protocol.tex, sec/results_decomp.tex.
- Old sec/*.tex moved to sec/archive/ (old intro restored from git HEAD). Old main.tex at sec/archive/main_working_2026-10-17.tex.
- Subagents running: methods brief (scratchpad/methods_brief.md), compute_table.py (num_compute.tex), mechanism_numbers.py (num_mechanism.tex).
- TODO: abstract, background, methods (+tab:methods), results_laws, results_more (5.3-5.6 + sec:pitfalls), discussion, related, appendix; add missing bib entries (flux1kreadev2025, bloc97ntk, radford2021clip, hessel2021clipscore, kirstain2023pickapic, xu2023imagereward, ke2021musiq, wang2023clipiqa, heusel2017fid, binkowski2018kid, dao2024fa2...).
## Progress 12:05
- Mechanism agent done: paper/tables/{mechanism_numbers.py,num_mechanism.tex,mechanism_summary.md} (macros \Nmech...). Key: Constancy onset step 20 (16-22, sigma .56), active 8/28 steps; TM material onset step 8; SEGA logit gain 1.00-1.61, 56% static/80% +time/20% prompt (7.3% late); probe width diag .47 prompt0 but -.03/-.04 prompts1,2; CR17 err .94; FLUX.1-dev top band -3.64 vs Krea .11.
- Methods brief at scratchpad/methods_brief.md (equations; temp_only_sbase differs from SEGA by allocation + clamp RMS lift + axis-0 gain 1.248 vs 1).
- Names aligned with figs/DESIGN_NOTES.md (SEGA (released), SEGA base + temp., SEGA + TM, Alias-T, SEGA (paper eqs.)).
- Written: abstract, intro, protocol, results_decomp, related, appendix (needs sec/appendix_impl.tex). TODO: background, methods, results_laws, results_more, discussion, appendix_impl; fix intro/abstract actuator claim (weak, prompt-dependent, poorly predictable; not 'anisotropic leaky'); fix 'last four steps' in results_decomp -> onset step 20.
## Progress 12:45
- All sections drafted (sec/: abstract, hero (page-1 minipage), intro, background, methods, protocol, results_decomp, results_laws, results_more, discussion, related, appendix, appendix_impl, appendix_compute stub). Builds: 23 pp, main text runs to p.12 + floats pushed past refs.
- Figures F1,F3-F8 exist (F2 pending: figs/f2_method.tex). F4 is 7.5in tall.
- Next (delegated to layout subagent): move F4/F6/F8 to appendix, add 5.7 pitfalls (sec:pitfalls) + appendix_compute from compute agent outputs, trim main to <=10 pp, page-1 hero, visual check every page, captions vs figures.
- Compute agent still running (paper/tables/compute_table.py -> tab_compute.tex, num_compute.tex).
## Progress 13:05
- Compute agent done: tab_compute.tex, num_compute.tex (37 macros). Corrections: FA2 forced = 1.54x cuDNN same node (not 26%); cuDNN GH200 188 s; GB200 93 s (2.0x); 1558 imgs, 79 GPU-h.
- Coordinator caveats (loading scheme, onset sigma macros, F7a NTK base, DyPE speckle row 8 only, Constancy share both 110% crop/68% full, seed-1 spec larger) forwarded to layout agent; make_tables.py now writes shx@ share macros for all arms.
- Layout agent running (trim to <=10 pp, page-1 hero, pitfalls 5.7, appendix_compute, captions from figs/CAPTIONS.md, visual pass).
## Progress 17:58 CDT
- Layout agent hit a rate limit at ~12:25 after editing abstract/background/hero/intro/methods/protocol/results_decomp/results_laws; resumed 17:58 with coordinator rule: main text only n=187 + n=55; Constancy n=41 and dev-only numbers only in new appendix app:partial.
- After it reports: verify build (0 overfull/undefined), grep prose rules, skim pages at low dpi, then final report to coordinator.
## Done 19:05 CDT
- paper/main.pdf: 22 pp, body 10 pp (refs p11), 0 overfull/undefined; standalone source zip /scratch/09032/saini_2/tmp/papercheck/constancy_rope_paper_src.zip compiles clean.
- Nothing committed (shared tree with figure agent). PLAN.md timing claims (26%, 2.2x) superseded by compute_table.py (1.54x, 2.01x).
