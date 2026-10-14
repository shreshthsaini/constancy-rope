"""Keep CPU checks small and write generated fixtures only to project scratch."""

import os
from pathlib import Path
import uuid

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/scratch/09032/saini_2/constancy_rope/cache/matplotlib")


def pytest_configure(config):
    if config.option.basetemp is None:
        parent = Path("/scratch/09032/saini_2/constancy_rope/test_runs")
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = parent / f"pytest_{uuid.uuid4().hex}"

