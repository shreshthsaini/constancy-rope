# Integrating the strengthening campaign into the paper

For the text owner. Final version, 2026-10-24 afternoon, written by the strengthening analyst after every block was
scored and analysed. Nothing here edits main.tex or a section you own. Numbers in this file are for orientation;
in the paper quote the macros from `tables/num_strengthen.tex`. Paste-ready findings with macro names:
`.agent/notes/strengthen/DIGEST.md`. Full numbers: `.agent/notes/strengthen/RESULTS.md`.

## Outcome in one look

| Test or block | Result |
| --- | --- |
| H1 (registered, pooled 337) | supported: ImageReward +0.009 [-0.020, +0.041], PickScore -0.017 [-0.059, +0.026], Holm p 0.0015 each |
| H1 on fresh 150 alone (descriptive) | not established: PickScore -0.073 [-0.133, -0.011] crosses -0.10 |
| H2 (registered, fresh 150) | supported: allocation adds +1.56 [+0.43, +2.72] full-frame MUSIQ, Holm p 0.0045 |
| Factorial (held-out 187) | temperature +0.594 [+0.506, +0.686] ImageReward (all of SEGA's +0.594 gain over NTK); base +0.027 [-0.003, +0.056]; no base or interaction test survives Holm |
| Sweep, seeds, fresh arms | no multiplier beats s^0.16; seeds 0-2 repeat the tie and the MUSIQ gap; only DyPE beats SEGA (crop MUSIQ, CLIP-IQA) |
| Controls (dev) | lattice is a checkpoint effect (12.0x, 7.4x at matched guidance); default shift: ImageReward -3.70 [-3.96, -3.42] |
| Qwen-Image | not run (OOM beside a co-resident process); no claim |

## Where things go

1. Preamble of `main.tex`, right after `\input{tables/num_results}`:
   `\InputIfFileExists{tables/num_strengthen}{}{}`
2. Body: `\input{sec/results_confirmation}` right after `\input{sec/results_more}`. The section is final and
   unguarded. It places its own floats: figures `fs_factorial.pdf` (`fig:fs-factorial`) and `fs_confirm.pdf`
   (`fig:fs-confirm`), and the two main-text tables through `\input{tables/float_strengthen_factorial}`
   (`tab:strengthen-factorial`) and `\input{tables/float_strengthen_fresh_tests}` (`tab:strengthen-fresh-tests`).
3. Appendix: `\input{tables/float_strengthen_appendix}` after `\input{sec/appendix}`. It is one lettered appendix
   section, "Confirmation campaign: full tables" (`app:confirm`), with an opening sentence and the floats
   `tab:strengthen-factorial-full`, `-factorial-means`, `-sweep` (plus `fig:fs-sweep`), `-fresh-arms`, `-seeds`,
   `-ckpt`, `-default-shift`. Change `\section` to `\subsection` there only by editing strengthen_tables.py.
4. A scratch copy with exactly these three edits builds cleanly with tectonic (no overfull box, no undefined
   reference from the new files; 35 pages). The section ran from page 9 to page 12, so the body grows by about 3.5
   pages; decide what moves to the appendix. Every table caption ends with a "Key result:" sentence.

## Placeholder comments to resolve (all in files you own)

| Placeholder | Replace with |
| --- | --- |
| `results_decomp.tex` `% STRENGTHEN FACTORIAL` | a pointer to `\cref{sec:confirm}` plus the revised base-and-temperature reading below |
| `results_decomp.tex` `% STRENGTHEN SWEEP` | one sentence: no multiplier beats $s^{0.16}$ (`\cref{tab:strengthen-sweep,fig:fs-sweep}`) |
| `results_laws.tex` `% STRENGTHEN SEEDS` and `% STRENGTHEN FRESH` | pointers to the seed and fresh paragraphs of `\cref{sec:confirm}` |
| `results_more.tex` `% STRENGTHEN CHECKPOINT` | see superseded sentence 7 |
| `results_more.tex` `% STRENGTHEN QWEN` | delete; the section's last paragraph reports that Qwen did not run |
| `background.tex` `% STRENGTHEN DEFAULT SHIFT` | see superseded sentence 8 |
| `protocol.tex` `% STRENGTHEN` | one paragraph: factorial arms, five fixed multipliers, fresh150 sampling (seed 2026 permutation, caption-only audit), seed means, Holm families (3, 80, 40, 50, 10) |
| `limitations.tex` `% STRENGTHEN` | H1 pooled with inspected prompts; fresh-only PickScore equivalence not established; Qwen not run; surrogate metrics only |

## Sentences the new results supersede (quoted as they stand)

1. `abstract.tex`: "\sega's base and global temperature alone recover all of its ImageReward gain over an NTK base
   (\shp{ir}, interval \shci{ir}) and tie the released method on three reward models and crop MUSIQ."
   The share is still right, but the factorial assigns it to the temperature: `\N{st@f@d@temp@ir}
   \N{st@f@ci@temp@ir}` of the `\N{st@f@tot@ir}` gain, base `\N{st@f@d@base@ir} \N{st@f@ci@base@ir}`. The tie is
   now a registered equivalence on the pooled prompts (H1 `\N{st@h@H1}`).
2. `abstract.tex`: "The allocation adds \hoa{alloc}{musiq} full-frame MUSIQ points alongside reduced excess
   fine-scale power, without an established causal link." Add that the gain replicates on fresh prompts (H2,
   `\N{st@h@d@musiq} \N{st@h@ci@musiq}`).
3. `intro.tex` paragraph 4: "Base and temperature carry most of the gain (\cref{fig:hero}): of \sega's gain over
   the \ntk{} base, they carry \shp{ir} of the ImageReward gain (interval \shci{ir}), \shp{pick} of PickScore and
   \shp{musiqc} of crop MUSIQ. \tempsb{} ties the released method on all three reward models, where a tie means that
   the paired interval covers zero." Name the temperature as the carrier and cite H1 instead of "ties".
4. `intro.tex` second bullet: "We show that \sega's base and global temperature recover \shp{ir} \shci{ir} of its
   ImageReward gain over an \ntk{} base, while its allocation adds \hoa{alloc}{musiq} full-frame MUSIQ points."
   Becomes: the global temperature carries the gain, the base adds nothing detectable, the allocation's MUSIQ gain
   replicates on fresh prompts.
5. `results_decomp.tex`: subsection title "Base and temperature do most of the work", paragraph lead "Base and
   temperature carry the reward-model gain", and "Of \sega's gain over \ntk, base and temperature thus carry
   \shp{ir} of the ImageReward gain (interval \shci{ir}), \shp{pick} of PickScore, \shp{clip} of CLIP and
   \shp{musiqc} of crop MUSIQ". Keep the bundle numbers and add one sentence that the factorial
   (`\cref{sec:confirm}`) attributes the bundle to the temperature.
6. `limitations.tex`: "\item \textbf{Bundled knobs.} No held-out arm separates the base and temperature changes of
   \tempsb{} relative to \ntk." This is no longer true; replace with the pooled-H1 and fresh-PickScore caveats.
7. `results_more.tex`: "Our port is therefore not a necessary cause of that failure, but the guidance difference
   prevents isolating the checkpoint." The matched-guidance control now isolates it: `\N{st@k@ratio@g3p5}$\times$
   \N{st@k@ci@g3p5}` at 3.5 and `\N{st@k@ratio@g4p5}$\times$ \N{st@k@ci@g4p5}` at 4.5.
8. `background.tex`: "This diagnosis is numerical, since we did not measure the default's effect on image quality,
   but comparisons that mix the two schedules confound position embedding with sampling." Now measured: ImageReward
   `\N{st@ds@d@def@ir} \N{st@ds@ci@def@ir}` on 8 development prompts (`\cref{tab:strengthen-default-shift}`).
9. `hero.tex` caption (a): "\tempsb{} minus \ntk{} (base\,+\,temp.) carries most of the gain". Still true as a
   bundle; the figure agent may split panel (a) into base, temperature and allocation to match `fig:fs-factorial`.

## Summary paragraph for the text editor

A registered follow-up confirms and sharpens the decomposition. On the 187 held-out prompts, a two-by-two factorial
of RoPE base and global logit temperature shows that the temperature carries released SEGA's whole ImageReward gain
over an NTK base (+0.594 [+0.506, +0.686] of +0.594), while SEGA's frequency base adds +0.027 [-0.003, +0.056] and
no base or interaction test survives Holm across ten metrics. On 337 pooled prompts, the base-plus-temperature
control is equivalent to released SEGA within the preregistered margins (ImageReward +0.009 [-0.020, +0.041],
PickScore -0.017 [-0.059, +0.026], Holm p 0.0015), although the pool reuses inspected prompts and the 150 fresh
prompts alone leave PickScore equivalence unestablished (-0.073 [-0.133, -0.011]). On those fresh prompts the per-band
allocation still adds +1.56 [+0.43, +2.72] full-frame MUSIQ (Holm p 0.0045), three seeds per held-out prompt repeat
both findings, and a fixed sweep finds no logit multiplier better than SEGA's $s^{0.16}$. Two development controls
show that the FLUX.1-dev lattice is a checkpoint effect at matched guidance and that the diffusers default flow shift
lowers ImageReward by 3.70 at 4K. No arm beats released SEGA on a reward model; DyPE's only wins are crop MUSIQ and
CLIP-IQA. The Qwen-Image transfer test did not run.

## Notes for the build

- CI order: new analyses bootstrap in manifest order; the old tables used lexical order (registered exception).
  Points agree, endpoints can differ in the last digit, e.g. allocation full MUSIQ `[+0.97, +3.25]` here versus
  `\hoci{alloc}{musiq}` = `[+0.96, +3.25]` in the old tables. Do not mix both versions of one contrast in a sentence.
- Pre-existing, not from these files: `sec/appendix_figs.tex` line 29 cites `\Cref{tab:papercode_scores}`, a label
  that exists nowhere, so the PDF prints ?? there.
- Never write that a method beats SEGA unless a Holm-marked interval says so; the only such cells are DyPE on crop
  MUSIQ and CLIP-IQA (fresh and held-out), together with Holm losses on both reward models.

## Regenerate (project root, OMP_NUM_THREADS=8, crope python)

```
scripts/strengthen_confirm.py run          # merge, registered analyses, CSV exports, RESULTS.md tables
paper/tables/strengthen_tables.py          # macros, tables, table floats, appendix file
paper/figs/strengthen/fs_factorial.py; paper/figs/strengthen/fs_sweep.py; paper/figs/strengthen/fs_confirm.py
```
