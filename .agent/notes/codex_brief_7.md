Camera-ready pass on the paper in /work/09032/saini_2/vista/Projects/Exploration/constancy_rope/paper (main.tex, sec/*.tex,
figs/*.py, tables/make_tables.py). Read paper/PLAN.md (honesty rules: no SOTA claim, no "beats SEGA" claim, every number from
data via scripts) and .agent/notes/fable_review_2026-10-18.md (senior review; its items 4 to 9 were applied in the working
tree, which is committed as the baseline). Another Codex process is idle-polling the GPU queue and will later ADD confirmation
results; do not touch anything outside paper/, and do not touch .agent/notes/prereg_strengthen.md or taskq.

The supervisor reviewed the current build (27 pages, body 11 pages) and found these defects. Fix all of them.

1. Figure 1b (figs/f1_hero.py) is unreadable: cramped per-metric axes, a clipped "0.77" label, hatched negative segments too
   small to read, overlapping intervals, and a caption that has to apologize for translated intervals. Redesign panel b as a
   dot-and-interval chart: for each of the four metrics, two labelled rows on one shared per-metric axis: "base + temperature"
   (SEGA base + temp. minus NTK, paired 95% CI) and "allocation" (SEGA released minus SEGA base + temp., paired 95% CI), a zero
   line, value labels at the dot, consistent colors from figs/crope_style.py, no hatching, no stacking, no translation.
   Numbers come from the same macros/CSVs as before (scratch eval_krea/heldout). Panel a may stay (crops fixed after viewing,
   selection rule stated in one short sentence). Caption: finding first, then at most three plain sentences of method facts.
   Iterate at least three rounds: render with paper/render_pages.sh at 160 dpi and look at the PNG each round.
2. Body length: Sections 1 to 7 must end on page 9 (references start on page 10). Do it by moving secondary material to the
   appendix (candidates: Table 2 stage-by-stage paper-vs-code table with a one-sentence summary left in Section 3; the
   per-law derivations in 3.1 to a short list plus appendix; parts of 5.5 pitfalls; related work tightened) and by cutting
   filler, never by removing caveats, CIs, Holm marks, or the honesty statements. Keep every figure currently in the main text
   except where moving it to the appendix is required for length; Figure 3 (forest) and Figure 4 (spectrum) stay in the main text.
3. Figure 2 (figs/f2_method.tex, TikZ): verify all text is at least 6 pt at the printed width, labels do not overlap, the three
   knobs and the two controllers are legible; rebuild if needed (tectonic on the standalone .tex, then the PDF is included).
4. Consistency pass across text, tables, figures: method names and colors per figs/crope_style.py; number formatting per
   metric; "tie" defined at first use; no sentence claims a method beats SEGA (released); dev-only numbers only in the
   appendix and labelled; cleveref references in order; no overfull boxes; no widows/orphans at page tops; references
   formatted consistently (arXiv entries uniform, no raw URLs for arXiv items; keep DOIs where present).
5. Spot-check 20 numbers in the abstract, intro and Section 5 against paper/tables/*.tex macros and scratch CSVs; fix any drift.
6. Build with ~/.local/bin/tectonic in paper/; render every page (paper/render_pages.sh main.pdf /scratch/09032/saini_2/tmp/
   cr7 110) and inspect each page image; fix layout defects; repeat until clean. Then refresh paper/dist/constancy_rope_paper.pdf
   and paper/dist/constancy_rope_paper_src.zip (zip must compile on its own in a fresh scratch directory; test it).

Constraints: shared CPU node, OMP_NUM_THREADS=8, own processes under 20; no ssh/sbatch/scancel/pkill/killall; never delete
scratch data; no em dashes (no --- or unicode em dash) or section symbol; avoid the words license, warrant, plainly. Run
figure scripts with /work/09032/saini_2/vista/library/uv-envs/crope/bin/python. You cannot commit; leave changes in the tree.
Final report under 300 words: page counts (body, references start, total), what moved to the appendix, figure 1b design,
number spot-check outcome, remaining known weaknesses.
