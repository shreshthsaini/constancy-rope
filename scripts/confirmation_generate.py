"""Generate one fixed chunk only after a valid third-diagnostic selection."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation_analysis import PROMPT_SHA256, selected_candidates, sha256
from crope.utils import atomic_json, image_stem, read_prompts, runtime_environment

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
NOTES = ROOT / '.agent/notes/continuation3'


def validated_chunk(method, start, end):
    if not (0 <= start < end <= 64) or end - start > 8:
        raise ValueError('Require a chunk of at most eight frozen confirmation prompts')
    selected = selected_candidates(json.loads((NOTES / 'dev/selection.json').read_text()))
    if method not in ('sega_official', *selected):
        raise ValueError('Cannot generate an unselected confirmation method')
    prompts = NOTES / 'confirmation64_prompts.jsonl'
    if sha256(prompts) != PROMPT_SHA256:
        raise ValueError('Frozen confirmation prompt manifest changed')
    frozen = json.loads((NOTES / 'frozen_manifest.json').read_text())
    for name, digest in frozen['source_sha256'].items():
        path = Path(name)
        if path.parent == ROOT / 'crope' or path.name == 'krea_profile_tm.npz':
            if sha256(path) != digest:
                raise ValueError(f'Frozen controller or profile changed: {path}')
    provenance = json.loads((NOTES / 'dev/generation_provenance.json').read_text())
    transformer = Path(os.environ['CROPE_TRANSFORMER'])
    if (str(transformer.resolve()) != provenance['transformer_snapshot'] or
            sha256(transformer / 'transformer/config.json') != provenance['transformer_config_sha256']):
        raise ValueError('Actual transformer differs from development')
    return read_prompts(prompts)[start:end], provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    args = parser.parse_args()
    prompts, provenance = validated_chunk(args.method, args.start, args.end)
    runtime_environment()
    import torch
    import wandb
    from crope.runtime import FluxExperiment, load_pipeline
    generation_root = SCRATCH / f'out_krea/confirm64_{args.method}'
    generation_root.mkdir(parents=True, exist_ok=True)
    manifest = generation_root / 'generation_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError('Existing confirmation generation provenance differs')
    atomic_json(manifest, provenance)
    hardware = {'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
                'job_id': os.environ.get('SLURM_JOB_ID'),
                'host': os.uname().nodename, 'start': args.start, 'end': args.end,
                'method': args.method, 'prompt_ids': [r['id'] for r in prompts]}
    runs = generation_root / 'generation_runs'
    runs.mkdir(exist_ok=True)
    run_path = runs / f'{hardware["job_id"]}_{args.start}_{args.end}.json'
    atomic_json(run_path, {**hardware, 'complete': False})
    run = wandb.init(project='constancy-rope', group='confirmation64', job_type='generation',
                     name=f'{args.method}-{args.start:02d}-{args.end:02d}',
                     config={**hardware, **provenance, 'prompt_sha256': PROMPT_SHA256,
                             'purpose': 'Frozen confirmation; no further selection'})
    experiment = None
    try:
        pipe = load_pipeline('cuda', False, False)
        profile = SCRATCH / 'calib/krea_profile_tm.npz' if args.method == 'sega_ctm_lag' else None
        experiment = FluxExperiment(pipe, args.method, (4096, 4096), profile=profile, eta=1.0)
        out = generation_root / args.method / '4096x4096'
        for index, row in enumerate(prompts):
            experiment.sample(row, out, seed=0, steps=28, guidance=4.5)
            record = json.loads((out / f"{image_stem(row['id'], 0)}.json").read_text())
            run.log({'chunk_images_complete': index + 1, 'prompt_id': row['id'],
                     'seconds_per_image': record['seconds'], 'image_sha256': record['png_sha256']})
        atomic_json(run_path, {**hardware, 'complete': True})
    finally:
        if experiment is not None:
            experiment.close()
        run.finish()


if __name__ == '__main__':
    main()
