#!/usr/bin/env python3
"""Frozen Qwen-Image transfer controls around the released SEGA implementation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import atomic_json, file_sha256, image_stem, read_prompts, runtime_environment

SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
OFFICIAL = SCRATCH / 'third_party/sega'
OFFICIAL_COMMIT = 'ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e'
MODEL_REVISION = '75e0b4be04f60ec59a75f475837eced720f823b6'
MODEL = Path('/scratch/09032/saini_2/tmp/hf/hub/models--Qwen--Qwen-Image/snapshots') / MODEL_REVISION
SUFFIX = ', Ultra HD, 4K, cinematic composition.'
ARMS = ('qwen_base', 'qwen_sega', 'qwen_temp_only')


def source_provenance():
    commit = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(OFFICIAL), 'status', '--porcelain'], text=True).strip()
    if commit != OFFICIAL_COMMIT or status:
        raise ValueError(f'Official SEGA source differs from pin: {commit}; {status[:200]}')
    paths = [OFFICIAL / 'qwen_sega' / name for name in
             ('run_qwen.py', 'pipeline_qwenimage.py', 'transformer_qwenimage.py')]
    paths += [Path(__file__), ROOT / 'crope/utils.py']
    paths += [MODEL / name for name in ('model_index.json', 'transformer/config.json',
                                       'scheduler/scheduler_config.json')]
    return {'official_commit': commit, 'source_sha256': {str(p): file_sha256(p) for p in paths},
            'versions': {name: importlib.metadata.version(name) for name in
                         ('torch', 'diffusers', 'transformers', 'accelerate')}}


def load_official(module_name):
    key = '_strengthen_qwen_' + module_name
    if key in sys.modules:
        return sys.modules[key]
    path = OFFICIAL / 'qwen_sega' / (module_name + '.py')
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def arm_control(transformer, arm):
    """Only change RoPE: native factors, released SEGA, or uniform spatial gain."""
    if arm not in ARMS:
        raise ValueError(arm)
    rope = transformer.pos_embed
    original_forward, original_alpha = rope.forward, rope.mscale_alpha
    if arm == 'qwen_base':
        def native_forward(*args, **kwargs):
            kwargs.update(ntk_factor_h=1.0, ntk_factor_w=1.0, mscale_spread=None,
                          energy_profile_h=None, energy_profile_w=None)
            return original_forward(*args, **kwargs)
        rope.forward = native_forward
    elif arm == 'qwen_temp_only':
        rope.mscale_alpha = 0.0
    try:
        yield
    finally:
        rope.forward = original_forward
        rope.mscale_alpha = original_alpha


def fixed_config(arm, row, provenance, *, smoke=False, cpu_offload=False):
    if arm not in ARMS:
        raise ValueError(arm)
    return {'schema': 'qwen-strengthen-v1', 'method': arm, 'prompt_id': str(row['id']),
            'prompt': row['prompt'], 'effective_prompt': row['prompt'] + SUFFIX,
            'negative_prompt': '', 'res': '4096x4096', 'steps': 2 if smoke else 40,
            'seed': 0, 'resolved_seed': 0, 'generator_device': 'cuda',
            'generator_policy': 'reset CUDA generator to seed0 for each prompt and arm',
            'true_cfg_scale': 4.0, 'guidance_scale': None, 'max_sequence_length': 512,
            'dtype': 'bfloat16', 'model': 'Qwen/Qwen-Image', 'model_revision': MODEL_REVISION,
            'model_path': str(MODEL), 'target_resolution': None, 'training_resolution': 1024,
            'rope_theta': 10000, 'scale_rope': True, 'spatial_rope_dim': 56,
            'spatial_ntk_factor': 1.0 if arm == 'qwen_base' else 4**(112/54)/(1+.1*math.log(4)),
            'spatial_base_qk_amplitude': 1.0 if arm == 'qwen_base' else 4**.08,
            'spatial_uniform_logit_gain': 1.0 if arm == 'qwen_base' else 4**.16,
            'frame_qk_amplitude': 1.0, 'mscale_alpha': .15 if arm == 'qwen_sega' else 0.0,
            'mscale_beta': 1.5, 'mscale_min': 1.0, 'spread_range': [0.0, 1.0], 'spread_alpha': 1.5,
            'released_width_allocation_reused_on_height': arm == 'qwen_sega',
            'schedule': 'released pipeline upper sequence anchor=image_seq_len',
            'mu': 0.9, 'shift_terminal': 0.02, 'use_dynamic_shifting': True,
            'time_shift_type': 'exponential',
            'vae_tiling': True, 'cpu_offload': cpu_offload, 'smoke': smoke,
            'provenance': provenance}


def verify_existing(png, metadata, config):
    if not png.exists() and not metadata.exists():
        return False
    if not png.is_file() or not metadata.is_file():
        raise ValueError(f'Preserve incomplete pair before retry: {png}')
    saved = json.loads(metadata.read_text())
    if saved.get('complete') is not True or saved.get('config') != config:
        raise ValueError(f'Existing Qwen identity/config differs: {metadata}')
    if saved.get('png_sha256') != file_sha256(png):
        raise ValueError(f'Existing Qwen image hash differs: {png}')
    from PIL import Image
    with Image.open(png) as image:
        if image.size != (4096, 4096):
            raise ValueError(f'Wrong Qwen image size: {png}')
        image.verify()
    return True


def cpu_checks():
    """Tiny random networks and exact rotary controls; no pretrained weights."""
    import torch
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenEmbedRope as NativeRope, QwenImageTransformer2DModel as NativeTransformer)
    torch.set_num_threads(8)
    module = load_official('transformer_qwenimage')
    load_official('pipeline_qwenimage')
    # Defaults use centered coordinates. Check the complete native image/text rotary tensors.
    official_rope = module.QwenEmbedRope(10000, [16, 56, 56], scale_rope=True,
                                      base_mscale_coefficient=.08, training_res_pixels=1024)
    native_rope = NativeRope(10000, [16, 56, 56], scale_rope=True)
    from types import SimpleNamespace
    holder = SimpleNamespace(pos_embed=official_rope)
    kwargs = dict(video_fhw=[[(1, 256, 256)]], max_txt_seq_len=7, device=torch.device('cpu'),
                  ntk_factor_h=4**(112/54)/(1+.1*math.log(4)),
                  ntk_factor_w=4**(112/54)/(1+.1*math.log(4)), mscale_spread=.7,
                  energy_profile_h=torch.linspace(1, 5, 128), energy_profile_w=torch.linspace(5, 1, 128),
                  target_res_h=4096, target_res_w=4096)
    with arm_control(holder, 'qwen_base'):
        base = official_rope(**kwargs)
    native = native_rope(kwargs['video_fhw'], max_txt_seq_len=7, device=torch.device('cpu'))
    rotary_error = max(float((a-b).abs().max()) for a,b in zip(base,native))
    if rotary_error != 0:
        raise RuntimeError(f'Native rotary parity failed: {rotary_error}')
    with arm_control(holder, 'qwen_temp_only'):
        uniform = official_rope(**kwargs)
        first = official_rope.compute_mscale(kwargs['ntk_factor_h'], kwargs['ntk_factor_w'], .7,
                         kwargs['energy_profile_h'], kwargs['energy_profile_w'], 4096, 4096, torch.device('cpu'))
        second = official_rope.compute_mscale(kwargs['ntk_factor_h'], kwargs['ntk_factor_w'], .1,
                         torch.rand(128), torch.rand(128), 4096, 4096, torch.device('cpu'))
    if not torch.equal(first,second) or not torch.all(first[:8] == 1):
        raise RuntimeError('Uniform control depends on spectral energy or changes frame gain')
    if not torch.allclose(first[8:], torch.full_like(first[8:], 4**.08), atol=0, rtol=1e-7):
        raise RuntimeError('Uniform spatial gain differs from released base temperature')
    released = official_rope(**kwargs)
    allocation_effect = max(float((a-b).abs().max()) for a,b in zip(released,uniform))
    if not allocation_effect > 0:
        raise RuntimeError('Released allocation did not change high-resolution rotary tensors')
    with arm_control(holder, 'qwen_base'):
        native_again = official_rope(**kwargs)
    if any(not torch.equal(a,b) for a,b in zip(base,native_again)):
        raise RuntimeError('Changing arms left a stale positional cache')
    # Check complete transformer forward parity with identical tiny random weights.
    torch.manual_seed(2026)
    tiny = dict(patch_size=2, in_channels=16, out_channels=4, num_layers=1,
                attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                axes_dims_rope=(4,6,6))
    ordinary = NativeTransformer(**tiny).eval()
    controlled = module.QwenImageTransformer2DModel(**tiny, training_resolution=16).eval()
    controlled.load_state_dict(ordinary.state_dict(), strict=True)
    inputs = dict(hidden_states=torch.randn(1,16,16), encoder_hidden_states=torch.randn(1,5,32),
                  encoder_hidden_states_mask=torch.ones(1,5,dtype=torch.bool),
                  timestep=torch.tensor([.5]), img_shapes=[[(1,4,4)]], return_dict=False)
    with torch.inference_mode(), arm_control(controlled,'qwen_base'):
        reference=ordinary(**inputs)[0]
        actual=controlled(**inputs)[0]
    forward_error=float((actual-reference).abs().max())
    if not torch.allclose(actual,reference,rtol=1e-5,atol=1e-6):
        raise RuntimeError(f'Native full-transformer parity failed: {forward_error}')
    return {'passed': True, 'native_rotary_max_abs_error': rotary_error,
            'tiny_native_transformer_max_abs_error': forward_error,
            'uniform_energy_independent': True, 'frame_gain': 1.0,
            'native_after_other_arms_exact': True,
            'spatial_qk_amplitude': 4**.08, 'spatial_logit_gain': 4**.16,
            'released_allocation_changes_rotary_max_abs': allocation_effect,
            'pretrained_weights_loaded': False, 'gpu_used': False}


def finish_smoke(args, row, provenance):
    records=[]
    for arm in ARMS:
        png=args.out/arm/'4096x4096'/(image_stem(row['id'],0)+'.png')
        config=fixed_config(arm,row,provenance,smoke=True,cpu_offload=args.cpu_offload)
        if not verify_existing(png,png.with_suffix('.json'),config):
            raise RuntimeError('Incomplete smoke output')
        saved=json.loads(png.with_suffix('.json').read_text())
        if saved.get('finite_steps') != [True,True]:
            raise RuntimeError('Smoke lacks two finite-step checks')
        records.append({'arm':arm,'path':str(png),'sha256':saved['png_sha256'],
                        'seconds':saved['seconds'],'sidecar_sha256':file_sha256(png.with_suffix('.json'))})
    atomic_json(args.smoke_receipt,{'passed':True,'mode':'two-step 4K three-arm execution only',
                'model_revision':MODEL_REVISION,'provenance':provenance,'records':records,
                'cpu_offload':args.cpu_offload,'quality_evidence':False})


def generate(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen generation requires an allocated GPU worker')
    torch.set_num_threads(8)
    provenance = source_provenance()
    rows = read_prompts(args.prompts)
    if args.mode == 'smoke':
        if args.start != 0 or args.end != 1:
            raise ValueError('Smoke uses exactly dev row0')
        arms = ARMS
    else:
        if args.arm is None or not 0 <= args.start < args.end <= len(rows):
            raise ValueError('Specify one arm and valid prompt range')
        arms = (args.arm,)
        smoke_receipt = json.loads(args.smoke_receipt.read_text())
        if not smoke_receipt.get('passed') or smoke_receipt.get('model_revision') != MODEL_REVISION:
            raise ValueError('Successful matching GPU smoke required before campaign generation')
        if smoke_receipt.get('provenance') != provenance:
            raise ValueError('Sources differ from successful GPU smoke')
        if smoke_receipt.get('cpu_offload') != args.cpu_offload:
            raise ValueError('Campaign memory policy differs from successful GPU smoke')
    missing=[]
    for row in rows[args.start:args.end]:
        for arm in arms:
            config=fixed_config(arm,row,provenance,smoke=args.mode=='smoke',cpu_offload=args.cpu_offload)
            stem=image_stem(row['id'],0)
            out=args.out / arm / '4096x4096'
            png,sidecar=out/(stem+'.png'),out/(stem+'.json')
            if not verify_existing(png,sidecar,config):
                missing.append((row,arm,config,png,sidecar))
    if not missing:
        if args.mode == 'smoke':
            finish_smoke(args,rows[0],provenance)
        print('All requested Qwen samples verified complete.',flush=True)
        return
    module=load_official('transformer_qwenimage')
    pipeline=load_official('pipeline_qwenimage')
    transformer=module.QwenImageTransformer2DModel.from_pretrained(str(MODEL),subfolder='transformer',
                  torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True,
                  training_resolution=1024,spread_min=0.,spread_max=1.,spread_alpha=1.5,
                  mscale_alpha=.15,mscale_beta=1.5,mscale_min=1.,
                  base_mscale_formula='power_res',base_mscale_coefficient=.08)
    pipe=pipeline.QwenImagePipeline.from_pretrained(str(MODEL),transformer=transformer,
                  torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.vae.enable_tiling()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    import wandb
    if not os.environ.get('WANDB_API_KEY'):
        os.environ.setdefault('WANDB_MODE','offline')
    run=wandb.init(project=os.environ.get('WANDB_PROJECT','constancy-rope'),dir=os.environ['WANDB_DIR'],
                   job_type='qwen-'+args.mode,config={'first_sample':missing[0][2],'n':len(missing)})
    records=[]
    try:
        for row,arm,config,png,sidecar in missing:
            finite_steps=[]
            def check_step(_pipe,index,timestep,values):
                finite=bool(torch.isfinite(values['latents']).all())
                finite_steps.append(finite)
                if not finite:
                    raise RuntimeError(f'Nonfinite Qwen latents at step{index}')
                return values
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0=time.perf_counter()
            with torch.inference_mode(), arm_control(transformer,arm):
                image=pipe(prompt=config['effective_prompt'],negative_prompt='',height=4096,width=4096,
                           num_inference_steps=config['steps'],true_cfg_scale=4.0,
                           generator=torch.Generator('cuda').manual_seed(0),max_sequence_length=512,
                           callback_on_step_end=check_step).images[0]
            torch.cuda.synchronize()
            seconds=time.perf_counter()-t0
            if image.size != (4096,4096) or len(finite_steps) != config['steps']:
                raise RuntimeError('Qwen image size or step count differs from config')
            png.parent.mkdir(parents=True,exist_ok=True)
            temp=png.with_name(png.name+f'.tmp.{os.getpid()}')
            image.save(temp,format='PNG')
            os.replace(temp,png)
            record={'complete':True,'config':config,'png_sha256':file_sha256(png),'seconds':seconds,
                    'peak_memory_bytes':torch.cuda.max_memory_allocated(),'finite_steps':finite_steps,
                    'scheduler':dict(pipe.scheduler.config),'sigmas':pipe.scheduler.sigmas.float().cpu().tolist(),
                    'timesteps':pipe.scheduler.timesteps.float().cpu().tolist(),'wandb_run_id':run.id,
                    'created_utc':datetime.now(timezone.utc).isoformat(),'host':os.uname().nodename,
                    'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name()}
            atomic_json(sidecar,record)
            records.append({'arm':arm,'path':str(png),'sha256':record['png_sha256'],'seconds':seconds})
            run.log({'seconds':seconds,'peak_memory_bytes':record['peak_memory_bytes']})
            print(json.dumps(records[-1]),flush=True)
        if args.mode == 'smoke':
            finish_smoke(args,rows[0],provenance)
    finally:
        run.finish()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','smoke','generate'))
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--prompts',type=Path,default=ROOT/'prompts/fresh150.jsonl')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=1)
    parser.add_argument('--out',type=Path,default=SCRATCH/'out_qwen/fresh')
    parser.add_argument('--cpu-offload',action='store_true')
    parser.add_argument('--smoke-receipt',type=Path,default=SCRATCH/'diag/qwen_smoke/passed.json')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    runtime_environment()
    if args.mode=='check':
        report={'checks':cpu_checks(),'provenance':source_provenance()}
        if args.report:
            atomic_json(args.report,report)
        print(json.dumps(report,indent=2))
    else:
        generate(args)


if __name__=='__main__':
    main()
