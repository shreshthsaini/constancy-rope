"""Two-forward, read-only capture of accepted CR18 attention operators.

The baseline is uninstrumented. Capture observes the actual SDPA arguments and
return value inside the unchanged gained_attention implementation. No allocator
is recomputed and no attention output is replaced. GPU execution requires an
explicit frozen protocol; importing helpers and running unit tests is CPU-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

from scripts.observer_pretrained_audit import (
    BASE, KREA, read_json, require, sha256, tensor_hash, validate_protocol as validate_cr18_protocol,
)
from scripts.observer_pretrained_report import difference_stats, valid_digest, verify_gain, verify_tensor
from crope.utils import atomic_json, runtime_environment

SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
CR18_PROTOCOL_SHA256 = 'b98de615bdb3c475b38f355f35b13fbce50240f614aaac9492ebf803a77374da'
CR18_REPORT_SHA256 = '127ef1c2a2b5b27e590a1d80ee7baead59bf6ec9375c59a4f5616dcbef84df49'
CAPTURE = {
    'attention_call_indices_zero_based': [0, 18, 56],
    'heads_zero_based': [0, 12],
    'image_query_grid_coordinates': [[y, x] for y in (40, 104, 168, 232) for x in (40, 104, 168, 232)],
}
REQUIRED_SOURCES = {
    'scripts/attention_operator_capture.py', 'tests/test_attention_operator_capture.py',
    'scripts/observer_pretrained_audit.py', 'scripts/observer_pretrained_report.py',
    'crope/gain.py', 'crope/rope.py', 'crope/runtime.py', 'crope/utils.py',
}


def validate_protocol(protocol):
    fixed = {'schema_version': 1, 'status': 'frozen', 'development_prompt_indices': [0, 1],
             'forwards_per_prompt': 2, 'forward_order': ['baseline', 'capture'],
             'resolution': 4096, 'grid': [256, 256], 'sigma': .5, 'guidance': 4.5,
             'text_tokens': 512, 'attention_heads': 24, 'head_dim': 128,
             'expected_attention_calls': 57, 'dtype': 'bfloat16', 'capture': CAPTURE}
    for key, expected in fixed.items():
        require(protocol.get(key) == expected, f'Capture protocol {key} mismatch')
    for key in ('accepted_cr18_report', 'accepted_cr18_receipt', 'cr18_protocol', 'witness_retention'):
        entry = protocol.get(key, {})
        require(set(entry) == {'path', 'sha256'} and isinstance(entry['path'], str)
                and Path(entry['path']).is_absolute() and valid_digest(entry['sha256']),
                f'Invalid pinned {key}')
    require(protocol['cr18_protocol']['sha256'] == CR18_PROTOCOL_SHA256 and
            protocol['accepted_cr18_report']['sha256'] == CR18_REPORT_SHA256,
            'Only the accepted CR18 protocol/report may supply this diagnostic')
    sources = protocol.get('source_sha256', {})
    require(REQUIRED_SOURCES <= set(sources), 'Capture sources are not frozen')
    require(all(not Path(name).is_absolute() and '..' not in Path(name).parts and valid_digest(digest)
                for name, digest in sources.items()), 'Invalid source path/hash')


def pinned_json(entry):
    require(sha256(entry['path']) == entry['sha256'], f'Pinned JSON changed: {entry["path"]}')
    return read_json(entry['path'])


def retained_path(receipt, original_path, digest):
    matches = [record for record in receipt['files'] if record['source'] == original_path]
    require(len(matches) == 1 and matches[0]['sha256'] == digest, 'Retained witness mapping mismatch')
    path = Path(matches[0]['preserved'])
    require(path.is_absolute() and path.is_relative_to(Path(receipt['archive_root'])),
            'Retained witness is outside declared archive')
    require(path.is_file() and sha256(path) == digest, 'Retained witness file changed')
    return path


def load_accepted_inputs(protocol, prompt_index):
    """Validate immutable sources and durable CR18 witnesses before model loading."""
    validate_protocol(protocol)
    require(prompt_index in (0, 1), 'Only the two existing development prompts are permitted')
    for name, digest in protocol['source_sha256'].items():
        require(sha256(ROOT / name) == digest, f'Frozen capture source changed: {name}')
    old_protocol = pinned_json(protocol['cr18_protocol'])
    validate_cr18_protocol(old_protocol)
    require(len(old_protocol['source_sha256']) == 16, 'Expected sixteen frozen CR18 source entries')
    for name, digest in old_protocol['source_sha256'].items():
        require(sha256(ROOT / name) == digest, f'Frozen CR18 source changed: {name}')
    report = pinned_json(protocol['accepted_cr18_report'])
    receipt = pinned_json(protocol['accepted_cr18_receipt'])
    retained = pinned_json(protocol['witness_retention'])
    require(receipt['report_sha256'] == protocol['accepted_cr18_report']['sha256'], 'CR18 receipt/report mismatch')
    require(report['protocol_sha256'] == CR18_PROTOCOL_SHA256 and
            report['inputs']['protocol'] == old_protocol, 'CR18 embedded protocol mismatch')
    require(report['completed_prompts'] == [0, 1] and report['total_transformer_forwards'] == 10 and
            report['automatic_promotion'] is False, 'CR18 campaign not accepted')
    verified = {}
    for index in (0, 1):
        label = f'prompt{index}'
        raw, tensors = report['raw_json'][label], report['tensor_verification'][label]
        manifest, completion = raw['manifest.json'], raw['completion.json']
        require(completion == raw['validation.json'] and completion['exact_repeat_equal'] is True and
                completion['completed_forwards'] == 5, 'CR18 completion mismatch')
        require(manifest['prompt_index'] == index and manifest['protocol_sha256'] == CR18_PROTOCOL_SHA256 and
                manifest['source_sha256'] == old_protocol['source_sha256'] and
                manifest['model'] == old_protocol['model'], 'CR18 manifest provenance mismatch')
        require(manifest['model']['transformer_revision'] == KREA and
                manifest['model']['base_revision'] == BASE, 'Unexpected pretrained revisions')
        state_receipt = tensors['input']
        path = retained_path(retained, state_receipt['path'], state_receipt['file_sha256'])
        state, _ = verify_tensor(path, 'input', (1, 65536, 64), torch.bfloat16,
            file_digest=manifest['input_file_sha256'], tensor_digest=manifest['state']['input_sha256'])
        original, repeated = raw['original_g0.json'], raw['repeat_original_g0.json']
        require(original['output_sha256'] == repeated['output_sha256'], 'CR18 original replay was not byte equal')
        gains = verify_gain(original['gain'])
        require(original['gain'] == repeated['gain'] and original['attention_calls'] == 57 and
                original['consumed_gain_bfloat16_sha256'] == tensor_hash(gains.bfloat16()),
                'CR18 original gain/consumption receipt mismatch')
        output_receipt = tensors['outputs']['original_g0']
        output_path = retained_path(retained, output_receipt['path'], output_receipt['file_sha256'])
        old_output, _ = verify_tensor(output_path, 'output', (1, 65536, 64), torch.bfloat16,
            file_digest=original['output_file_sha256'], tensor_digest=original['output_sha256'])
        verified[index] = (state.clone(), gains, old_output.clone(), manifest)
    return verified[prompt_index]


def query_indices(grid, coordinates, text_tokens):
    require(len(coordinates) > 0 and len({tuple(c) for c in coordinates}) == len(coordinates),
            'Query coordinates must be nonempty and unique')
    require(all(len(c) == 2 and all(type(x) is int for x in c) and
                0 <= c[0] < grid[0] and 0 <= c[1] < grid[1] for c in coordinates),
            'Query outside image grid')
    image = torch.tensor([y * grid[1] + x for y, x in coordinates], dtype=torch.int64)
    return image, image + text_tokens


def tensor_receipt(value):
    require(bool(torch.isfinite(value).all()), 'Captured tensor is nonfinite')
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': tensor_hash(value)}


class AttentionCapture:
    """Single-threaded scoped observer; tensors saved in B,H,S,D SDPA layout."""

    def __init__(self, *, expected_gain, grid, text_tokens, total_heads, head_dim,
                 selected_heads, coordinates, selected_calls, expected_calls, sink):
        self.gain = expected_gain.detach().clone()
        require(self.gain.shape == (head_dim,) and self.gain.dtype == torch.float32 and
                bool(torch.isfinite(self.gain).all()), 'Expected finite FP32 head gain')
        require(len(selected_heads) > 0 and len(set(selected_heads)) == len(selected_heads) and
                all(type(h) is int and 0 <= h < total_heads for h in selected_heads), 'Invalid selected heads')
        require(len(selected_calls) > 0 and len(set(selected_calls)) == len(selected_calls) and
                all(type(c) is int and 0 <= c < expected_calls for c in selected_calls), 'Invalid selected calls')
        self.image_ids, self.query_ids = query_indices(grid, coordinates, text_tokens)
        self.head_ids = torch.tensor(selected_heads, dtype=torch.int64)
        self.sequence = text_tokens + grid[0] * grid[1]
        self.total_heads, self.head_dim = total_heads, head_dim
        self.selected_calls, self.expected_calls = set(selected_calls), expected_calls
        self.sink, self.calls, self.sdpa_calls = sink, 0, 0
        self.captured = []
        self.records = {}

    def _outer_check(self, query, key, value, scale, mask):
        expected = (1, self.sequence, self.total_heads, self.head_dim)
        require(mask is None, 'Attention mask is outside capture target')
        require(all(tuple(t.shape) == expected and t.dtype == torch.bfloat16 and t.device == query.device
                    for t in (query, key, value)), 'Unexpected Q/K/V shape, dtype or device')
        require(scale.dtype == torch.float32 and scale.shape == self.gain.shape and
                torch.equal(scale, self.gain.to(scale.device)), 'Attention gain changed')

    @contextmanager
    def installed(self, gain_module):
        original_attention, original_sdpa = gain_module.gained_attention, F.scaled_dot_product_attention

        def attention(query, key, value, qk_scale, attention_mask=None):
            self._outer_check(query, key, value, qk_scale, attention_mask)
            index = self.calls
            require(index < self.expected_calls, 'Too many attention calls')
            self.calls += 1
            inner_count = 0
            payload = None

            def sdpa(q, k, v, *args, **kwargs):
                nonlocal inner_count, payload
                inner_count += 1
                self.sdpa_calls += 1
                require(inner_count == 1 and not args and
                        set(kwargs) == {'attn_mask', 'dropout_p', 'is_causal'} and
                        kwargs['attn_mask'] is None and kwargs['dropout_p'] == 0.0 and
                        kwargs['is_causal'] is False, 'Unexpected SDPA arguments or call count')
                expected = (1, self.total_heads, self.sequence, self.head_dim)
                require(all(tuple(t.shape) == expected and t.dtype == torch.bfloat16 and t.device == query.device
                            for t in (q, k, v)), 'Unexpected consumed SDPA geometry')
                if index in self.selected_calls:
                    heads, queries = self.head_ids.to(q.device), self.query_ids.to(q.device)
                    # Copy the actual arguments passed after gain multiplication, not reconstructed Q/K.
                    payload = {
                        'query': q.index_select(2, queries).index_select(1, heads).detach().cpu().contiguous(),
                        'key': k.index_select(1, heads).detach().cpu().contiguous(),
                        'value': v.index_select(1, heads).detach().cpu().contiguous(),
                        'head_indices': self.head_ids.clone(), 'query_joint_indices': self.query_ids.clone(),
                        'query_image_indices': self.image_ids.clone(),
                        'key_joint_indices': torch.arange(self.sequence),
                        'gain_bfloat16': qk_scale.bfloat16().detach().cpu().contiguous(),
                    }
                output = original_sdpa(q, k, v, *args, **kwargs)
                require(tuple(output.shape) == expected and output.dtype == torch.bfloat16,
                        'Unexpected SDPA output geometry')
                if payload is not None:
                    payload['sdpa_output'] = output.index_select(2, queries).index_select(1, heads).detach().cpu().contiguous()
                return output

            require(F.scaled_dot_product_attention is original_sdpa, 'SDPA wrapper already installed')
            F.scaled_dot_product_attention = sdpa
            try:
                result = original_attention(query, key, value, qk_scale, attention_mask)
            finally:
                F.scaled_dot_product_attention = original_sdpa
            require(inner_count == 1, 'Expected one SDPA call per attention block')
            if payload is not None:
                self.records[index] = {name: tensor_receipt(tensor) for name, tensor in payload.items()}
                self.sink(index, payload, self.records[index])
                self.captured.append(index)
            return result

        gain_module.gained_attention = attention
        try:
            yield self
        finally:
            gain_module.gained_attention = original_attention
            F.scaled_dot_product_attention = original_sdpa

    def validate_complete(self):
        require(self.calls == self.sdpa_calls == self.expected_calls, 'Attention call count mismatch')
        require(set(self.captured) == self.selected_calls and len(self.captured) == len(self.selected_calls),
                'Missing or duplicate capture')


def validate_replay(baseline, captured):
    require(baseline.dtype == captured.dtype == torch.bfloat16 and baseline.shape == captured.shape and
            bool(torch.isfinite(baseline).all()) and bool(torch.isfinite(captured).all()),
            'Invalid replay outputs')
    require(tensor_hash(baseline) == tensor_hash(captured), 'Baseline/capture outputs are not byte equal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--prompt-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime_environment()
    protocol = read_json(args.protocol)
    state, gain, prior_output, accepted = load_accepted_inputs(protocol, args.prompt_index)
    require(torch.cuda.is_available(), 'Allocated CUDA GPU required; no CPU model execution')
    require(not args.out.exists() and args.out.resolve().is_relative_to(SCRATCH), 'Use a fresh scratch output directory')
    model = accepted['model']
    transformer_path = Path(os.environ['CROPE_TRANSFORMER'])
    hub = Path(os.environ['HF_HOME']) / 'hub/models--black-forest-labs--FLUX.1-dev'
    base_snapshot = hub / 'snapshots' / BASE
    require(transformer_path.name == KREA and sha256(transformer_path / 'transformer/config.json') ==
            model['transformer_config_sha256'], 'Transformer snapshot/configuration mismatch')
    require((hub / 'refs/main').read_text().strip() == BASE and sha256(base_snapshot / 'model_index.json') ==
            model['base_model_index_sha256'], 'Base snapshot/configuration mismatch')
    import diffusers
    require(torch.__version__ == accepted['torch'] and torch.version.cuda == accepted['cuda'] and
            diffusers.__version__ == accepted['diffusers'], 'Software differs from accepted CR18')
    from crope.runtime import load_pipeline
    from crope.rope import FluxRoPE
    import crope.gain as gain_module
    import wandb
    args.out.mkdir(parents=True)
    manifest = {'purpose': protocol.get('purpose', 'Bounded attention capture'), 'prompt_index': args.prompt_index,
        'prompt': accepted['prompt'], 'protocol_sha256': sha256(args.protocol), 'source_sha256': protocol['source_sha256'],
        'accepted_cr18_report_sha256': CR18_REPORT_SHA256, 'cr18_protocol_sha256': CR18_PROTOCOL_SHA256,
        'model': model, 'input_sha256': tensor_hash(state), 'gain_float32_sha256': tensor_hash(gain),
        'gain_bfloat16_sha256': tensor_hash(gain.bfloat16()), 'capture': CAPTURE,
        'layout': 'B,H,S,D; joint keys are text first then image raster order',
        'sdpa_target': {'scale': 'default 1/sqrt(128)', 'dropout_p': 0., 'is_causal': False, 'attn_mask': None},
        'forward_order': ['baseline', 'capture'], 'forward_budget': 2,
        'count_scope': '57 observed outer/inner calls on capture; baseline has 57 architecture blocks and no wrapper',
        'on_trajectory': False, 'efficacy_evaluated': False, 'torch': torch.__version__,
        'diffusers': diffusers.__version__, 'cuda': torch.version.cuda, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip()}
    atomic_json(args.out / 'manifest.json', manifest)
    run = None
    try:
        atomic_json(args.out / 'progress.json', {'status': 'loading_model', 'completed_forwards': 0})
        mode = os.environ.get('WANDB_MODE', 'online' if os.environ.get('WANDB_API_KEY') else 'offline')
        try:
            run = wandb.init(project='constancy-rope', group='attention-operator', job_type='capture',
                name=f'attention-operator-{args.prompt_index}', dir=str(SCRATCH / 'wandb'), config=manifest, mode=mode)
        except Exception:
            if mode == 'offline':
                raise
            run = wandb.init(project='constancy-rope', group='attention-operator', job_type='capture',
                name=f'attention-operator-{args.prompt_index}', dir=str(SCRATCH / 'wandb'), config=manifest, mode='offline')
        pipe = load_pipeline('cuda', False, False)
        require(Path(pipe.config['_name_or_path']).resolve() == base_snapshot.resolve() and
                (hub / 'refs/main').read_text().strip() == BASE, 'Loaded base revision changed')
        pipe.transformer.eval()
        require(len(pipe.transformer.transformer_blocks) == 19 and len(pipe.transformer.single_transformer_blocks) == 38 and
                pipe.transformer.config.in_channels == 64 and tuple(pipe.transformer.config.axes_dims_rope) == (16, 56, 56),
                'Unexpected FLUX geometry')
        require(sorted({str(p.dtype) for p in pipe.transformer.parameters()}) == accepted['model_parameter_dtypes'],
                'Parameter dtypes differ from CR18')
        pipe.transformer.pos_embed = FluxRoPE(axes_dims=(16, 56, 56), method='sega_official', grid=(256, 256))
        controller = SimpleNamespace(qk_scale=gain.cuda())
        gain_module.install_gain(pipe.transformer, controller)
        with torch.inference_mode():
            prompt_embeds, pooled, txt_ids = pipe.encode_prompt(prompt=accepted['prompt']['prompt'], prompt_2=None,
                device=torch.device('cuda'), num_images_per_prompt=1, max_sequence_length=512)
            img_ids = pipe._prepare_latent_image_ids(1, 256, 256, torch.device('cuda'), torch.bfloat16)
            conditioning = {'encoder_hidden_states': prompt_embeds, 'pooled_projections': pooled,
                            'txt_ids': txt_ids, 'img_ids': img_ids}
            require({k: tensor_hash(v) for k, v in conditioning.items()} == accepted['conditioning_sha256'],
                    'Conditioning hashes differ from accepted CR18')
            require(prompt_embeds.shape[1] == 512, 'Unexpected text token count')
            kwargs = dict(hidden_states=state.cuda(), **conditioning,
                timestep=torch.tensor([.5], device='cuda', dtype=torch.bfloat16),
                guidance=torch.tensor([4.5], device='cuda', dtype=torch.float32), return_dict=False)
            manifest['conditioning_sha256'] = accepted['conditioning_sha256']
            atomic_json(args.out / 'manifest.json', manifest)
            run.config.update(manifest, allow_val_change=True)
            saved = {}

            def sink(index, payload, records):
                path = args.out / f'attention_{index:02d}.pt'
                torch.save(payload, path)
                saved[str(index)] = {'path': str(path.resolve()), 'file_sha256': sha256(path), 'tensors': records}
                atomic_json(args.out / f'attention_{index:02d}.json', saved[str(index)])

            capture = AttentionCapture(expected_gain=gain, grid=(256, 256), text_tokens=512, total_heads=24,
                head_dim=128, selected_heads=CAPTURE['heads_zero_based'],
                coordinates=CAPTURE['image_query_grid_coordinates'],
                selected_calls=CAPTURE['attention_call_indices_zero_based'], expected_calls=57, sink=sink)
            outputs, timings = {}, {}
            for index, arm in enumerate(('baseline', 'capture')):
                atomic_json(args.out / 'progress.json', {'status': 'forward_running', 'arm': arm, 'completed_forwards': index})
                torch.cuda.synchronize()
                started = time.perf_counter()
                if arm == 'baseline':
                    with pipe.transformer.cache_context('cond'):
                        value = pipe.transformer(**kwargs)[0]
                else:
                    with capture.installed(gain_module), pipe.transformer.cache_context('cond'):
                        value = pipe.transformer(**kwargs)[0]
                    capture.validate_complete()
                torch.cuda.synchronize()
                timings[arm] = time.perf_counter() - started
                require(value.shape == (1, 65536, 64) and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()),
                        'Invalid transformer output')
                outputs[arm] = value.cpu()
                torch.save({'output': outputs[arm]}, args.out / f'{arm}_output.pt')
                atomic_json(args.out / 'progress.json', {'status': 'forward_completed', 'arm': arm, 'completed_forwards': index+1})
                run.log({'completed_forwards': index+1, 'forward_seconds_including_capture': timings[arm]})
            comparison = {'within_job': difference_stats(outputs['capture'], outputs['baseline']),
                          'prior_cr18': difference_stats(outputs['baseline'], prior_output),
                          'within_job_byte_equal': tensor_hash(outputs['capture']) == tensor_hash(outputs['baseline']),
                          'prior_cr18_byte_equal': tensor_hash(outputs['baseline']) == tensor_hash(prior_output)}
            atomic_json(args.out / 'replay_validation.json', comparison)
            validate_replay(outputs['baseline'], outputs['capture'])
            completion = {'completed_forwards': 2, 'captured_attention_calls': capture.calls,
                'captured_sdpa_calls': capture.sdpa_calls, 'captures': saved, 'replay': comparison,
                'outputs': {arm: {'file_sha256': sha256(args.out / f'{arm}_output.pt'),
                                 **tensor_receipt(value)} for arm, value in outputs.items()},
                'seconds_including_capture': timings, 'on_trajectory': False, 'efficacy_evaluated': False,
                'interpretation': 'Within-job byte replay is required. Across-node prior CR18 output equality is descriptive only. Capture timing includes copies/hashes/I/O and is not a speed benchmark.'}
            atomic_json(args.out / 'completion.json', completion)
            atomic_json(args.out / 'progress.json', {'status': 'completed', 'completed_forwards': 2})
    except Exception as error:
        atomic_json(args.out / 'failure.json', {'error_type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main()
