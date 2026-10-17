"""Freeze the separately tested six-image pilot; refuse to overwrite a protocol."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import atomic_json, file_sha256
from scripts.crossmodal_phase_pilot import example_protocol, validate_dependencies
from scripts.native_comparator import read_json, require


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha256(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'.agent/notes/crossmodal_phase_protocol.json')
    a = p.parse_args()
    require(not a.out.exists(), 'Refuse to overwrite a frozen protocol')
    protocol = example_protocol(ROOT/'.agent/notes/native_comparator_protocol.json')
    sources = ['scripts/freeze_crossmodal_phase.py', 'scripts/crossmodal_phase_scores.py',
        'tests/test_crossmodal_phase_scores.py', 'scripts/eval.py', 'scripts/imagereward_score.py',
        'slurm/crossmodal_phase_task.sh', 'slurm/crossmodal_phase_fleet.sh',
        'slurm/fleet_crossmodal_phase_gh.sbatch']
    sources += [str(path.relative_to(ROOT)) for path in sorted((ROOT/'crope/metrics').glob('*.py'))]
    protocol['source_sha256'].update({name: file_sha256(ROOT/name) for name in sources})
    protocol.update(created_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        design=pin(ROOT/'.agent/notes/post_cr20_mechanism_design.json'),
        prior_art=pin(ROOT/'.agent/notes/post_cr20_prior_art.json'),
        review_plan=pin(ROOT/'.agent/notes/crossmodal_phase_review_plan.json'),
        scoring_reference=pin(ROOT/'.agent/notes/crossmodal_phase_scoring_reference.json'),
        imagereward_files=read_json(ROOT/'.agent/notes/crossmodal_phase_ir_pins.json'),
        total_full_images=6, total_native_smoke_transformer_forwards=32,
        total_full_resolution_transformer_forwards=168,
        compute_envelope={'nodes': 2, 'maximum_minutes': 40, 'central_minutes': 20,
                          'range_minutes': [15, 30], 'scope': 'Includes loading, smoke, generation and scoring; unbenchmarked estimate.'},
        continuation='No automatic promotion, parameter search, held-out access or fresh64 use. Validate and retain all six outputs before interpretation.',
        backend_limit='Explicit cuDNN only. Actual GPU kernel and native checks must pass before any4K generation.',
        timing_limit='Unnormalized control has an extra image-attention pass; sequential arms do not form a warmup-matched speed benchmark.')
    for index in (0, 1):
        validate_dependencies(protocol, index)
    atomic_json(a.out, protocol)
    print(str(a.out.resolve()), file_sha256(a.out))


if __name__ == '__main__':
    main()
