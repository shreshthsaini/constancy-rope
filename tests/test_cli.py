"""Direct script entry points must not require models for help."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script", ["sample", "calibrate", "probe_mtf", "eval", "prompts", "enqueue", "diagnose_schedule"])
def test_script_help(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / f"{script}.py"), "--help"],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
