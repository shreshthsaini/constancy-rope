#!/usr/bin/env python3
"""Matched checkpoint/guidance diagnostic using the unmodified official DyPE code."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import atomic_json, file_sha256, image_stem, read_prompts, runtime_environment

SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
HUB = Path('/scratch/09032/saini_2/tmp/hf/hub')
OFFICIAL_COMMIT = '436fe612fb7695fa2d5dc32d63390c048ba6033c'
DEV_REVISION = '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'
KREA_REVISION = '8162a9c7b05a641be098422bf2fcf335615c2f28'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, choices=('dev', 'krea'))
    p.add_argument('--guidance', required=True, type=float, choices=(3.5, 4.5))
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=28)
    p.add_argument('--prompts', type=Path, default=SCRATCH / 'prompts/aesthetic4k_eval.jsonl')
    p.add_argument('--official-root', type=Path, default=SCRATCH / 'third_party/DyPE')
    p.add_argument('--out', type=Path, default=SCRATCH / 'diag/ckpt_matrix')
    p.add_argument('--validate-only', action='store_true', help='Check sources/settings without importing torch or loading weights')
    return p


def method_name(checkpoint, guidance):
    return f'official_dype_{checkpoint}_g{guidance:.1f}'.replace('.', 'p')


def model_paths(checkpoint):
    base = HUB / 'models--black-forest-labs--FLUX.1-dev/snapshots' / DEV_REVISION
    transformer = (base if checkpoint == 'dev' else
                   HUB / 'models--black-forest-labs--FLUX.1-Krea-dev/snapshots' / KREA_REVISION)
    return base, transformer


def source_provenance(official_root):
    """Pin upstream identity and content, including untracked Python changes."""
    official_root = Path(official_root).resolve()
    commit = subprocess.check_output(['git', '-C', str(official_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f'Official DyPE commit differs: {commit}')
    status = subprocess.check_output(['git', '-C', str(official_root), 'status', '--porcelain'], text=True).strip()
    if status:
        raise ValueError(f'Official DyPE worktree is modified: {status[:500]}')
    files = sorted((official_root / 'flux').rglob('*.py'))
    for required in ('pipeline_flux.py', 'transformer_flux.py'):
        if official_root / 'flux' / required not in files:
            raise ValueError(f'Missing official source: {required}')
    return {'official_root': str(official_root), 'official_commit': commit,
            'official_python_sha256': {str(p.relative_to(official_root)): file_sha256(p) for p in files},
            'wrapper_sha256': file_sha256(__file__)}


def build_config(args, row, provenance, versions):
    base, transformer = model_paths(args.checkpoint)
    return {
        'schema': 'official-dype-checkpoint-matrix-v1',
        'method': method_name(args.checkpoint, args.guidance),
        'prompt_id': str(row['id']), 'prompt': row['prompt'], 'seed': args.seed,
        'generator': 'torch.Generator(device=cuda).manual_seed(seed), reset per prompt',
        'resolved_seed': args.seed, 'res': [4096, 4096], 'steps': args.steps,
        'guidance': args.guidance, 'checkpoint': args.checkpoint,
        'base_model': str(base), 'base_revision': DEV_REVISION,
        'transformer_model': str(transformer),
        'transformer_revision': DEV_REVISION if args.checkpoint == 'dev' else KREA_REVISION,
        'dtype': 'bfloat16', 'dype': True, 'rope_method': 'yarn',
        'official_yarn_dynamic_exponent': 2.0, 'official_yarn_kappa_coefficient': 1.0,
        'official_yarn_beta': [1.25, 0.75], 'official_yarn_gamma': [16.0, 2.0],
        'official_yarn_dynamic_rule': 'beta and gamma thresholds multiplied by timestep**2',
        'max_sequence_length': 512, 'vae_tiling': True,
        'schedule_policy': 'unmodified official DyPE: max_seq_len=image_seq_len',
        'expected_mu': 1.15, 'cpu_offload': False,
        'attention_backend': 'official implementation and current torch defaults',
        'prompt_manifest_sha256': file_sha256(args.prompts),
        'provenance': provenance, 'versions': versions,
    }


def completed_sample(png, metadata, config):
    """A complete matching pair is reusable; preserve and reject other existing files."""
    png, metadata = Path(png), Path(metadata)
    if not png.exists() and not metadata.exists():
        return False
    if not png.is_file() or not metadata.is_file():
        raise ValueError(f'Incomplete existing pair; preserve/move aside before rerun: {png}')
    saved = json.loads(metadata.read_text())
    if saved.get('complete') is not True or saved.get('config') != config:
        raise ValueError(f'Existing sample configuration differs: {metadata}')
    if saved.get('png_sha256') != file_sha256(png):
        raise ValueError(f'Existing PNG hash differs: {png}')
    from PIL import Image
    with Image.open(png) as image:
        if image.size != (4096, 4096):
            raise ValueError(f'Existing PNG dimensions differ: {png}')
        image.verify()
    return True


def validate_inputs(args):
    if not 0 <= args.start < args.end <= 8:
        raise ValueError('Checkpoint diagnostic only allows dev rows 0 through 7')
    if args.seed != 42 or args.steps != 28:
        raise ValueError('Preregistered checkpoint diagnostic requires seed 42 and 28 steps')
    prompts = read_prompts(args.prompts)
    if len(prompts) < 8:
        raise ValueError('Prompt manifest must include all 8 dev prompts')
    base, transformer = model_paths(args.checkpoint)
    for path in (base / 'model_index.json', base / 'scheduler/scheduler_config.json',
                 transformer / 'transformer/config.json'):
        if not path.is_file():
            raise ValueError(f'Missing cached model metadata: {path}')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'diffusers', 'transformers')}
    provenance = source_provenance(args.official_root)
    return [(i, row, build_config(args, row, provenance, versions))
            for i, row in enumerate(prompts) if args.start <= i < args.end]


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    entries = validate_inputs(args)
    if args.validate_only:
        print(json.dumps({'validated': True, 'n': len(entries), 'example_config': entries[0][2]}, indent=2))
        return 0
    out = args.out / method_name(args.checkpoint, args.guidance) / '4096x4096'
    work = []
    for index, row, config in entries:
        stem = image_stem(row['id'], args.seed)
        png, metadata = out / f'{stem}.png', out / f'{stem}.json'
        if not completed_sample(png, metadata, config):
            work.append((index, row, config, png, metadata))
    if not work:
        print('All requested diagnostic images are complete and verified.', flush=True)
        return 0

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic must run on an allocated GPU worker')
    sys.path.insert(0, str(args.official_root.resolve()))
    from flux.transformer_flux import FluxTransformer2DModel
    from flux.pipeline_flux import FluxPipeline
    import flux.pipeline_flux as official_pipeline
    import flux.transformer_flux as official_transformer
    for module in (official_pipeline, official_transformer):
        if not Path(module.__file__).resolve().is_relative_to(args.official_root.resolve()):
            raise RuntimeError(f'Imported unofficial module: {module.__file__}')
    torch.set_num_threads(8)
    base, transformer = model_paths(args.checkpoint)
    tr = FluxTransformer2DModel.from_pretrained(str(transformer), subfolder='transformer',
                                                torch_dtype=torch.bfloat16, dype=True, method='yarn',
                                                local_files_only=True)
    pipe = FluxPipeline.from_pretrained(str(base), transformer=tr, torch_dtype=torch.bfloat16,
                                         local_files_only=True).to('cuda')
    pipe.vae.enable_tiling()
    out.mkdir(parents=True, exist_ok=True)
    import wandb
    if not os.environ.get('WANDB_API_KEY'):
        os.environ.setdefault('WANDB_MODE', 'offline')
    run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'constancy-rope'),
                     dir=os.environ['WANDB_DIR'], job_type='checkpoint-guidance-diagnostic',
                     config={'sample': entries[0][2], 'start': args.start, 'end': args.end})
    try:
        for index, row, config, png, metadata in work:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.inference_mode():
                image = pipe(row['prompt'], height=4096, width=4096,
                             guidance_scale=args.guidance,
                             generator=torch.Generator('cuda').manual_seed(args.seed),
                             num_inference_steps=args.steps, max_sequence_length=512).images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if image.size != (4096, 4096):
                raise RuntimeError(f'Wrong image dimensions: {image.size}')
            temporary = png.with_name(png.name + f'.tmp.{os.getpid()}')
            image.save(temporary, format='PNG')
            os.replace(temporary, png)
            sigmas = pipe.scheduler.sigmas.detach().float().cpu().tolist()
            receipt = {'complete': True, 'config': config, 'png_sha256': file_sha256(png),
                       'seconds': elapsed, 'peak_memory_bytes': torch.cuda.max_memory_allocated(),
                       'scheduler': dict(pipe.scheduler.config), 'sigmas': sigmas,
                       'timesteps': pipe.scheduler.timesteps.detach().float().cpu().tolist(),
                       'prompt_row': index, 'created_utc': datetime.now(timezone.utc).isoformat(),
                       'host': os.uname().nodename, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                       'wandb_run_id': run.id, 'gpu': torch.cuda.get_device_name()}
            atomic_json(metadata, receipt)
            run.log({'prompt_row': index, 'seconds': elapsed,
                     'peak_memory_bytes': receipt['peak_memory_bytes']})
            print(json.dumps({'complete': str(png), 'seconds': elapsed}), flush=True)
    finally:
        run.finish()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
