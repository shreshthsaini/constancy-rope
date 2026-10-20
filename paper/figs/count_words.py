"""Count the text inside a figure PDF, the way DESIGN_NOTES.md reports it.

Usage (PyMuPDF lives in the pdfrender env):
  PYTHONPATH=/work/09032/saini_2/vista/library/uv-envs/pdfrender/lib/python3.11/site-packages \
  /home1/09032/saini_2/.local/share/uv/python/cpython-3.11.13-linux-aarch64-gnu/bin/python3.11 -s \
  paper/figs/count_words.py FIG.pdf [FIG2.pdf ...]

Tokens come from page.get_text("words"). English words: tokens with two or more letters and no
digit (math labels such as "ln" are listed so they can be set aside by hand). Numerals: tokens
with a digit. Symbols: everything else (single letters, Greek, arrows, operators).
"""
import re
import sys

import fitz

for path in sys.argv[1:]:
    words, nums, syms = [], [], []
    for page in fitz.open(path):
        for w in page.get_text("words"):
            tok = w[4].strip()
            if not tok:
                continue
            if re.search(r"\d", tok):
                nums.append(tok)
            elif len(re.findall(r"[A-Za-z]", tok)) >= 2:
                words.append(tok)
            else:
                syms.append(tok)
    print(f"{path}: {len(words)} English words, {len(nums)} numerals, {len(syms)} symbols")
    print("  words:", " ".join(words))
    print("  numerals:", " ".join(nums))
    print("  symbols:", " ".join(syms))
