# Final gate review, v3 (2026-10-24)

Reviewed: `paper/main.pdf` (34 pages, body pages 1 to 10, references 11 to 13, appendix 14 to 34), every page
rendered at 110 dpi (`/scratch/09032/saini_2/tmp/fable_v3/`), pages 1, 3, 7 and 8 at 160 dpi, against
`~/.claude/skills/gold-standard-paper/SKILL.md` Section 8, `paper/PLAN.md`, `.agent/notes/strengthen/DIGEST.md`,
the strengthening CSVs in `/scratch/09032/saini_2/constancy_rope/eval_krea/strengthen/`, the preregistration and
its amendment receipt, the 2026-10-18 review and the text agent's self-score in `HANDOVER.md`. No files edited.

## Verdict

Ships under the rule after the eight fixes below, all of which are one-pass edits. My rubric mean is 1.23 with no
zero (same mean as the self-score, with different items at 2 and 1). The one item I would hold the paper on is not a
rubric item: the abstract's "but no reward" and the undisclosed preregistration amendment (fixes 1 and 2).

## 1. Rubric (Section 8, scored 0/1/2)

| # | Item | Mine | Self | Justification |
| :-- | :-- | :-: | :-: | :-- |
| 1 | Abstract 130 to 200 words, headline number, URL | 1 | 1 | 165 to 168 words, factorial number with interval, GitHub URL. No closing implication sentence (previous review item 9, still open). |
| 2 | Hero under abstract, >= 0.9 width, panel-walking caption | 2 | 2 | 0.98 width; bars carry intervals and Holm stars, crops at identical native coordinates, selection rule in caption. Above the gold heroes. |
| 3 | At most two footnotes, administrative | 1 | 1 | None. |
| 4 | Question by end of para 1; motivating float beside para 2 | 1 | 1 | Question closes para 1. Table 1 wraptable is a knobs table, not a motivating picture; it does serve the "package comparison" argument. |
| 5 | "We introduce" para; numbers in para 4; two or three "We" bullets | 1 | 1 | Para 3 defines the control arm precisely; para 4 is all numbers; three "We" bullets, last carries 0.594. |
| 6 | Median sentence under 22, none over 35 | 1 | 1 | Self-reported median 20, max 32; spot-read of pages 2 and 7 agrees. Not re-measured by script. |
| 7 | Method diagram: width, 4 hues, 15 to 30 words, lettered panels, data, appendix zoom | 1 | 1 | Figure 2 meets every element (bridge photo flows through, F.1 zoom cited, about 25 words). Held at 1: the ln g panel is unreadable at print size (the "bars" for SEGA base + temp. read as green stripes), and the float splits the contribution bullets across pages 2 and 3. |
| 8 | Equations introduced by a clause, "where" line | 1 | 1 | Eq. (1) and (2), (3) all do. |
| 9 | Boxed side-by-side algorithms, Input, Output, appendix pointers | 1 | 1 | Algorithms 1 and 2 at the top of page 4, syntax colored, comments point to App. F, F.1 and Sec. 5.4. |
| 10 | Setup run-ins with counts, steps, compute hours | 2 | 2 | Eight run-ins; 79 GPU-hours, 1558 images, 28 steps; the registered-confirmation paragraph with five Holm families is beyond the gold papers. |
| 11 | Main table: booktabs, arrows, shading, bold best, Key result | 2 | 2 | Table 2 adds a paired-gain block with intervals and Holm stars. |
| 12 | Finding paragraphs: bold claim, before and after numbers | 1 | 1 | Every paragraph does it, with intervals. The 5.2 "Fresh prompts reproduce" paragraph is numbers without a reading sentence. |
| 13 | Ablation tables side by side, default shaded, appendix versions | 1 | 1 | Tables 4 and 5 at 0.45 and 0.5 width, C.3 and B.3 expand them. |
| 14 | Plots: palette, markers, value labels, shared legends, >= 6 pt | 1 | 1 | Figure text is outlined in the PDFs so sizes cannot be read from fonts; by eye nothing in Figures 1 to 4 is under 6.5 pt. Table direction triangles are 4.5 pt glyphs (pages 5 to 7, 19 to 28). Bars carry values; legends shared. |
| 15 | Qualitative grid, same prompt and noise, labeled rows | 1 | 1 | Figure A.2 and A.4 meet every element but sit in the appendix; the hero (b) is a 3-arm single-prompt strip. See Section 3 below: I disagree with the placement, not the score. |
| 16 | Confounds named; limitations with concrete scope | 2 | 2 | Six scoped bullets, pooled-prompt reuse, fresh PickScore, Qwen OOM, no human study. |
| 17 | Reproducibility statement mapping to appendices | 2 | 2 | Statement plus Appendix I with script names and tolerances. |
| 18 | Appendix: TOC, zoom, banded config table, expansions, random samples, prompts, pseudocode | 1 | 1 | All present except "random samples": A.2 and A.4 are the first three held-out rows, fixed before viewing and said so. Equivalent guarantee, different rule. |
| 19 | Floats within one page; >= 3 wraps; <= 60 percent float pages | 1 | 1 | Three wraps (Table 1, Figure 3, Table 6). Page 7 is about 55 percent float. Two defects that are not rubric elements: Figure 2 splits the bullet list; the Figure 3 wrapfigure's `[21]` line count overflows to page 8 and leaves three narrow orphan lines under Figure 4. |
| 20 | Captions above tables, below figures, bold leads | 1 | 1 | All. |
| 21 | Same colors across hero, plots, tables | 1 | 1 | Green temperature arm, blue SEGA, orange Constancy, gray DyPE, black NTK in every figure; table reference shading is the pale SEGA blue. Table direction triangles are neutral ink as PLAN requires. |
| 22 | No em dashes, no section symbol, nothing under 6 pt | 1 | 1 | grep clean on `sec/*.tex`; smallest TeX text is 6.1 pt math subscripts in Figure 2 and F.1; the 4.5 pt table triangles are the only sub-6 pt ink. |

Mean 27/22 = 1.23. No item at 0. Ships under the rule. Disagreements with the self-score: none on the numbers; I
hold item 7 at 1 for a different reason (readability of the gain panel, bullet split) and I would not have claimed
item 14 without measuring, which the self-score admits. The self-score's "Open" line is accurate.

## 2. Claims audit

Verified against the CSVs (`analysis_factorial_contrasts_holm.csv`, `analysis_fresh_confirmation_family.csv`,
`analysis_fresh_contrasts_holm.csv`, `analysis_seeds_contrasts_holm.csv`, `analysis_tsweep_means.csv`):

- Abstract: 187; temperature main effect +0.594 [+0.506, +0.686] (0.5938, 0.5057, 0.6860); base +0.027
  [-0.003, +0.056] (0.0267, -0.0026, 0.0566); 337 pooled, H1 ImageReward +0.009 [-0.020, +0.041] and PickScore
  -0.017 [-0.059, +0.026], both Holm p 0.0015; 150 fresh, H2 +1.56 [+0.43, +2.72], Holm p 0.0045. All match.
- Introduction: fresh PickScore -0.073 [-0.133, -0.011] (matches; Holm p 0.60, nominal only); temperature on NTK
  +0.603 and on SEGA base +0.585 (0.6026, 0.5849); 31 of 80 survive Holm (not recounted; Table C.1 is consistent);
  sweep means 0.476, 1.031, 1.148, 1.179, 1.017 and MUSIQ 41.97, 30.90 match; seeds +0.021 [-0.004, +0.044]
  matches; DyPE fresh crop MUSIQ +2.32 [+1.41, +3.25], CLIP-IQA +0.024, ImageReward -0.166, PickScore -0.38 match;
  NTK fresh -0.923 and SEGA base -0.875 match. Table C.5 correctly leaves the -0.07 PickScore cell unstarred.
- Arithmetic of "whole gain": sum of the two main effects equals SEGA base + temp. minus NTK exactly (0.620), and
  0.620 - 0.026 = 0.594 = SEGA (released) minus NTK, so Table 3's "add up" sentence is exact, not approximate.

Overclaims and missing caveats:

1. **Abstract, "but no reward" (overclaim).** On the fresh prompts the same comparison that fails PickScore
   equivalence is a nominal PickScore gain for the allocation: SEGA (released) minus SEGA base + temp. is +0.073
   [+0.011, +0.133], Holm p 0.60. "No reward" is false at the nominal level the paper uses everywhere else; the
   sentence contradicts the clause before it. Write "no Holm-surviving reward-model gain (fresh PickScore +0.07
   [+0.01, +0.13] nominal)" or "and no adjusted reward-model gain". The same qualifier belongs in the intro's "no
   detected reward-model gain" and the Section 5.1 run-in "The allocation adds full-frame MUSIQ, not reward", which
   is a held-out statement and should say so.
2. **Preregistration amendment not disclosed in the main text (missing caveat).** The original registration set
   H1 at +/-0.05 ImageReward on the fresh prompts alone. The amendment of 2026-10-19T04:11Z (receipt:
   `no_generation_queued: true`) widened it to +/-0.082 and pooled the 187 inspected prompts. This was legitimate and
   pre-generation, but Section 4 presents +/-0.082 and the 337-prompt pool as if original, and only Appendix I says
   "with amendments". Under the original rule the fresh-only ImageReward interval [-0.055, +0.035] would not have
   met the margin. One clause in Section 4 and one in 5.2: "margins and the pooled population were fixed by a dated
   amendment before any confirmation image existed; the original fresh-only +/-0.05 rule would not have been met".
   PLAN.md and the prereg's own text require this disclosure.
3. **Intro, "No multiplier in a fixed temperature sweep beats s^0.16" (scope).** True on ImageReward, PickScore and
   full-frame MUSIQ (Table 4's Key result says so); false on crop MUSIQ (1.45 wins by +6.19, Holm) and CLIP-IQA.
   Add "on a reward model or full-frame MUSIQ".
4. **Intro, lattice ratio 12.0x [7.8, 18.6].** Table C.7 labels this the historical peak-bin statistic, "biased
   upward"; the shared-bin value is 11.6x [7.5, 18.5]. Quote the shared bin, or say "peak-bin", and add "on eight
   dev prompts" to both this and the flow-shift number (-3.701), which the intro states without n.
5. **Title and "whole gain".** The 100 percent share is the coincidence that base (+0.027) and allocation (-0.026)
   cancel; the interval on the temperature effect spans 85 to 115 percent of the total. The intro's "+0.594 of
   +0.594" makes this honest; the abstract's "whole" is acceptable because the base number follows immediately. On
   fresh prompts NTK + temp. loses PickScore to released SEGA under Holm (-0.12), which 5.2 and the limitations state
   but the abstract does not; the existing "PickScore equivalence is not established" clause covers it adequately.
6. **Holm.** Used correctly throughout: stars only where the family is named, nominal intervals kept visible, the
   fresh PickScore cell unstarred. One wording risk: "Holm-adjusted equivalence p = 0.0015" is a TOST-style
   statement inside a three-test family; fine as written.
7. **Temperature does the work, mechanism.** Section 5.1's causal sentence is earned by the factorial. "Why a
   temperature" is labeled untested. Spectral paragraph says "consistent with" and "does not establish". Good.

Previous review items addressed: factorial arms (done, Table 3); H1 margin amendment (done, undisclosed in body);
matched-guidance checkpoint (done, Table C.7); "sharpens" removed and spectrum promoted (done); MUSIQ pipeline
(done); Holm marks (done); hero redrawn with absolute gains and matched prompt with selection rule (done);
cross-checkpoint probe moved to Appendix G, dev-origin and first-rows notes added (done); "104%" wording and
"tie" definition (done); abstract closing implication sentence (not done, it lives in Discussion); Figure 2
editorial labels removed (done); paper-vs-code line panel (done, Figure A.5); per-prompt G.1 matrices (done);
Nyquist as a dashed vertical (done); Qwen base-only arm and human study (not run, stated). Not followed: the
advice to keep the forest in the main text and to move a two-row qualitative grid into the body.

## 3. Judgment calls

**Forest plot to the appendix: right.** Table 2 and Figure 1a now carry the same 30 numbers, and the factorial
replaced the forest as the headline. Figure A.1 is well placed as the first appendix figure.

**Qualitative grid to the appendix: wrong for a gold-standard paper.** Both reference papers keep a qualitative
grid in the body, and this paper's central reader question ("does the allocation change what I see?") is visual.
The hero (b) shows three arms on one prompt; the reader never sees DyPE's speckle, Constancy-TM's loss or SEGA + TM
beside the reference. Figure 4 is the weakest use of body space: 0.3 page for 12 intervals that are all printed in
the text and in Table C.4, with a sign convention that changes between panels.

Proposed 10-page allocation (net change about zero):
- Page 3: place Figure 2 after the contribution bullets (`[b]` or after the Section 2 heading) so the list is not
  split.
- Page 8: Figure 4 becomes a 0.5-width wrapfigure with the two reward panels only, titled with the sign
  ("base + temp. minus SEGA"); the MUSIQ number stays in the text. Frees about 0.15 page.
- Page 9: a two-row version of Figure A.2 (rows 8 and 10, six arms, full frame plus one native crop each, about
  0.3 page) at the top of the page next to Section 5.3, caption "Figure A.2 shows the third prompt".
- Pages 9 to 10: fund the remaining 0.15 page by merging "Why a temperature" into "What to report", cutting the
  three-line "Spectral views" paragraph to one sentence inside "Rotary extrapolation", and compressing the
  Limitations bullets "Multiplicity" and "Ports" by one line each.

## 4. Figures and tables, main text

- **Figure 1.** Clear in three seconds; the small bars are the problem: the +0.027 and -0.026 whiskers collapse into
  the dashed line, so "base adds nothing" is read from the labels, not the plot. Fix: draw the total SEGA minus NTK
  gain as a thin black tick or bracket per metric, so "temperature equals the total" is visible without the caption.
- **Figure 2.** Handsome, real data, right word budget. The ln g panel cannot be decoded at print size (which marks
  are bars, circles, diamonds) and the float splits the bullet list. Fix: a three-glyph mini legend inside the panel
  and a placement after the bullets.
- **Figure 3.** Best plot in the body; gray band and Nyquist line carry the argument. The two TM lines are flat and
  overlap, and the wrapfigure overflows the page. Fix: set the wrapfigure line count to what the figure actually
  occupies (or `\vspace` the caption) so page 8 has no narrow orphan lines.
- **Figure 4.** Legible, but low information density and a sign flip between panels. Fix: 0.5-width wrap with the
  two reward panels, sign in the panel titles.
- **Table 1.** Exactly what the intro needs. Fix: none; optionally "Per-band" cells as glyph only, words to caption.
- **Table 2.** Exemplary. The "Share of base + temp." row (104 percent) now competes with the factorial framing
  and is explained three pages later. Fix: drop the share row from the body (Table B.2 keeps it) or move the
  "share above 100 percent" sentence into this caption.
- **Table 3.** Exactly right. Fix: add a "Total (SEGA minus NTK)" row so the "add up" sentence is checkable in
  place.
- **Table 4.** Good. Fix: Key result should add "crop MUSIQ and CLIP-IQA rise through 1.45" so the table is not read
  as "nothing changes".
- **Table 5.** Good. Fix: header "TM corr." to "Constancy-TM corr." to match every other label of that arm.
- **Table 6.** Good as a wraptable. Fix: none.
- **Algorithms 1 and 2.** Meet the bar. Fix: none.

## 5. Top 8 fixes, priority order

1. (text) Abstract and intro: replace "but no reward" / "no detected reward-model gain" with "no Holm-surviving
   reward-model gain (fresh PickScore +0.07 [+0.01, +0.13] nominal)"; add "(held out)" to the 5.1 run-in.
2. (text) Section 4 "Registered confirmation" and 5.2: one clause each disclosing the 2026-10-19 amendment
   (margin +/-0.05 to +/-0.082, fresh-only to pooled), that it preceded any confirmation image, and that the original
   rule would not have been met on fresh ImageReward.
3. (text) `sec/results_decomp.tex` line 13: fix the Figure 3 wrapfigure line count so the three orphan narrow lines
   at the top of page 8 disappear.
4. (text) Move Figure 2 so the contribution bullets are not split across pages 2 and 3.
5. (text) Intro: "beats s^0.16 on a reward model or full-frame MUSIQ"; lattice ratio as the shared-bin 11.6x [7.5,
   18.5] or labeled peak-bin; "on eight dev prompts" for the lattice and flow-shift numbers.
6. (figure) Two-row qualitative grid in Section 5.3, funded by shrinking Figure 4 to a 0.5-width two-panel wrap.
7. (figure) Figure 1a: total-gain tick per metric; Figure 2: three-glyph legend in the ln g panel.
8. (figure) Table triangles from 4.5 pt to 6 pt in `make_tables.py`; Table 5 header "Constancy-TM corr."; Table 3
   total row; abstract closing implication sentence ("Claims that credit spectral or content adaptivity should be
   tested against a base-plus-temperature control") if it fits under 200 words.
