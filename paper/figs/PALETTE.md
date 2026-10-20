# Paper palette

One palette for every figure and table. `crope_style.py` defines these values; tables must use
the same hex codes. Any one figure uses at most four hues; black and the neutral gray carry
baselines and text.

| Role | Arms | Color | 20% tint (diagram fill) | 10% tint (table row shading) |
| :-- | :-- | :-- | :-- | :-- |
| Method hue | SEGA (released) | `#0068A8` | `#CCE1EE` | `#E6F0F6` |
| Component: base and temperature | SEGA base + temp. | `#009870` | `#CCEAE2` | `#E6F5F1` |
| Component: per-band laws | Constancy, Constancy-TM, SEGA + TM, Alias-T | `#D06008` | `#F6DFCE` | `#FAEFE6` |
| Neutral gray | DyPE; reference marks | `#8C8C8C` | `#E8E8E8` | `#F4F4F4` |
| Black | NTK baseline; all text | `#1A1A1A` | | |
| Appendix only | SEGA (paper eqs.) | `#C878A0` | `#F4E4EC` | `#FAF2F6` |

Neutrals: secondary text `#595959`, hairlines `#BFBFBF`, gridlines `#E6E6E6`, gray bands and
prompt bands `#F2F2F2`. Zoom boxes on photos: red `#FF0000` (1 pt); a second box on the same
frame is cyan `#00B0F0`.

Markers and line styles (plots): NTK black circle, dashed; DyPE gray square, dashed;
SEGA (released) blue circle, solid; SEGA base + temp. green triangle, solid; Constancy-TM orange
diamond, solid; SEGA + TM orange down-triangle, dash-dot; Constancy orange circle, dotted.

Text naming a method is set in black; only marks carry the hue.

## LaTeX

```latex
\definecolor{cMethod}{HTML}{0068A8}  \definecolor{cMethodShade}{HTML}{E6F0F6}
\definecolor{cTemp}{HTML}{009870}    \definecolor{cTempShade}{HTML}{E6F5F1}
\definecolor{cLaw}{HTML}{D06008}     \definecolor{cLawShade}{HTML}{FAEFE6}
\definecolor{cGray}{HTML}{8C8C8C}    \definecolor{cGrayShade}{HTML}{F4F4F4}
\definecolor{cInk}{HTML}{1A1A1A}     \definecolor{cPaper}{HTML}{C878A0}
```

Table rule: shade the SEGA (released) row with `cMethodShade`; the default or main
configuration row of an ablation gets the same shade. Do not shade other arms.
