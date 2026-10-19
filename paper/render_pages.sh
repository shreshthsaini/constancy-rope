#!/bin/bash
# usage: paper/render_pages.sh FILE.pdf OUTDIR [DPI]   -> OUTDIR/page_NN.png (view them with an image reader)
PY=/home1/09032/saini_2/.local/share/uv/python/cpython-3.11.13-linux-aarch64-gnu/bin/python3.11
mkdir -p "$2"
PYTHONPATH=/work/09032/saini_2/vista/library/uv-envs/pdfrender/lib/python3.11/site-packages OMP_NUM_THREADS=8 $PY -s -c "
import sys, fitz
d = fitz.open(sys.argv[1])
for i, p in enumerate(d):
    p.get_pixmap(dpi=int(sys.argv[3])).save(f'{sys.argv[2]}/page_{i+1:02d}.png')
print(d.page_count, 'pages')" "$1" "$2" "${3:-110}"
