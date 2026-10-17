"""Generate the frozen eight-prompt diagnostic with per-image W&B records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crope.utils import read_prompts, runtime_environment, image_stem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=(
        'sega_official', 'sega_rms_uniform', 'sega_rms_normalized', 'sega_ctm_lag'))
    a = p.parse_args()
    runtime_environment()
    import wandb
    from crope.runtime import FluxExperiment, load_pipeline
    root = Path('/scratch/09032/saini_2/constancy_rope')
    transformer = Path(os.environ['CROPE_TRANSFORMER'])
    if transformer.name != '8162a9c7b05a641be098422bf2fcf335615c2f28':
        raise ValueError('Diagnostic requires the frozen Krea transformer revision')
    provenance = {'transformer_snapshot': str(transformer.resolve()),
                  'transformer_revision': transformer.name,
                  'transformer_config_sha256': hashlib.sha256((transformer / 'transformer/config.json').read_bytes()).hexdigest(),
                  'base_pipeline': 'black-forest-labs/FLUX.1-dev',
                  'note': 'Legacy image config model_revision describes the base pipeline, not the overridden transformer.'}
    prompts = read_prompts(root / 'prompts/aesthetic4k_eval.jsonl')[:8]
    out = root / f'out_krea/dev3_{a.method}' / a.method / '4096x4096'
    out.parent.parent.mkdir(parents=True, exist_ok=True)
    (out.parent.parent / 'generation_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    profile = root / 'calib/krea_profile_tm.npz' if a.method == 'sega_ctm_lag' else None
    run = wandb.init(project='constancy-rope', group='dev3-diagnostics',
                     name=f'dev3-{a.method}', job_type='generation',
                     config={'method': a.method, 'prompt_ids': [r['id'] for r in prompts],
                             'purpose': 'development diagnostic only', 'steps': 28,
                             'guidance': 4.5, 'seed': 0, 'eta': 1.0,
                             'slurm_job_id': os.environ.get('SLURM_JOB_ID'), **provenance})
    experiment = None
    try:
        pipe = load_pipeline('cuda', False, False)
        experiment = FluxExperiment(pipe, a.method, (4096, 4096), profile=profile, eta=1.0)
        for index, row in enumerate(prompts):
            experiment.sample(row, out, seed=0, steps=28, guidance=4.5)
            record = json.loads((out / f"{image_stem(row['id'], 0)}.json").read_text())
            run.log({'images_complete': index + 1, 'prompt_id': row['id'],
                     'seconds_per_image': record['seconds'],
                     'image_sha256': record['png_sha256']})
    finally:
        if experiment is not None:
            experiment.close()
        run.finish()


if __name__ == '__main__':
    main()
