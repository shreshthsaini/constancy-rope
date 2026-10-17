#!/bin/bash
#SBATCH -J CR-operator-v2
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --time-min=01:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/constancy_rope/watcher_cr14/controller_%j.log
set -euo pipefail
umask 077
cd /work/09032/saini_2/vista/Projects/Exploration/constancy_rope
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
unset CUDA_VISIBLE_DEVICES
/work/09032/saini_2/vista/library/uv-envs/dcc-watch/bin/python - <<'CHECK'
import json,hashlib
from pathlib import Path
p=Path('.agent/notes/controller_bootstrap_protocol.json')
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'c65e465aaa65c9e86067eecb4faf920f5d0d3b6986fd8174d55db4fbdd3dad9c', 'Controller protocol changed'
for f,h in json.loads(p.read_text())['source_sha256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, 'Controller source changed: '+f
CHECK
exec /work/09032/saini_2/vista/library/uv-envs/dcc-watch/bin/python -u scripts/controller_bootstrap.py --source-job 1073260 --codex /home1/09032/saini_2/.local/bin/codex
