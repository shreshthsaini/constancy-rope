"""Validate CR20 artifacts and prepare fixed, label-hidden review views."""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_comparator import (
    BASE, SCHEDULER_EXPECTATION, file_sha256, image_seed, pinned_json, read_json,
    require, validate_dependencies, validate_output,
)


@lru_cache(maxsize=8)
def verify_runtime_revision(revision, frozen_sources):
    """Runtime HEAD may add notes, but its frozen executable contents must agree."""
    require(isinstance(revision, str) and len(revision) == 40 and
            all(c in '0123456789abcdef' for c in revision), 'Invalid runtime code revision')
    for name, digest in frozen_sources:
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid frozen source path')
        content = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT)
        require(hashlib.sha256(content).hexdigest() == digest, 'Runtime commit source mismatch: '+name)


def expected_native_config(config, row, protocol, accepted, reference_config):
    """Derive semantic fields from pinned CR14 settings and frozen native algebra.

    Only two bookkeeping values come from the candidate: code revision, whose
    committed frozen files are verified, and ordering of an already-verified
    set of scheduler default-key names. Neither can supply semantic settings.
    """
    from crope.rope import FluxRoPE
    from crope.runtime import implementation_hash
    verify_runtime_revision(config.get('code_revision'), tuple(sorted(protocol['source_sha256'].items())))
    expected = copy.deepcopy(reference_config)
    expected.update(schema_version=1, method='sega_official', res='1024x1024', prompt_id=row['id'],
        prompt=row['prompt'], seed=0, derived_seed=image_seed(row['id'], 0), steps=28, guidance=4.5,
        model='black-forest-labs/FLUX.1-dev', model_revision=BASE, dtype='bfloat16', batch_size=1,
        attention='pytorch_sdpa_flash', torch=accepted['torch'], diffusers=accepted['diffusers'],
        code_revision=config['code_revision'], grid=[64, 64], eta=1., target='profile', temperature='logN',
        profile_sha256=None, static_profile_sha256=None, probe_band=None, probe_factor=1.5,
        rope=FluxRoPE(method='sega_official', grid=(64, 64)).metadata(),
        scale_choice='spatial axes per-side; scalar temperature geometric-mean side',
        side_scale=[1., 1.], area_scale=1., sega_sref_area_alternative=1.)
    require(expected['implementation_sha256'] == implementation_hash(), 'Frozen implementation fingerprint changed')
    expected['scheduler'].update(SCHEDULER_EXPECTATION)
    defaults = config.get('scheduler', {}).get('_use_default_values')
    reference_defaults = expected['scheduler'].get('_use_default_values')
    require(isinstance(defaults, list) and all(isinstance(x, str) for x in defaults)
            and len(defaults) == len(set(defaults)) and set(defaults) == set(reference_defaults),
            'Scheduler default-key set changed')
    expected['scheduler']['_use_default_values'] = defaults.copy()
    require(config == expected, 'Native config differs from independent frozen expectation')
    return expected


def validate_existing_configs(protocol_path, protocol_sha256, accepted_report, accepted_report_sha256, archive):
    """Read-only supplemental validation; never regenerate or rewrite archives."""
    require(file_sha256(protocol_path) == protocol_sha256, 'Protocol changed')
    require(file_sha256(accepted_report) == accepted_report_sha256, 'Accepted report changed')
    protocol, old = read_json(protocol_path), read_json(accepted_report)
    rows, accepted = validate_dependencies(protocol, 0, 4)
    require(old['accepted'] is True and old['images'] == 8 and old['protocol_sha256'] == protocol_sha256,
            'Unaccepted native report')
    archive = Path(archive)
    require(read_json(archive/'manifest.json') == old, 'Original archive manifest changed')
    retained = {entry['source']: entry for entry in old['retained']}
    checked = []
    for chunk, (start, end) in enumerate(protocol['chunks']):
        require(len(old['raw'][str(chunk)]['completion']['records']) == 4, 'Missing chunk records')
        manifest = old['raw'][str(chunk)]['manifest']
        for index, record in zip(range(start, end), old['raw'][str(chunk)]['completion']['records']):
            require(record['prompt_index'] == index, 'Native row order changed')
            path = Path(record['metadata'])
            require(file_sha256(path) == record['metadata_sha256'], 'Native metadata changed')
            entry = retained[str(path)]
            preserved = Path(entry['preserved'])
            require(preserved.is_relative_to(archive) and not preserved.is_symlink() and
                    file_sha256(preserved) == entry['sha256'] == record['metadata_sha256'], 'Archived metadata changed')
            config = read_json(preserved)['config']
            reference = pinned_json(protocol['selected_prompts'][index]['reference_metadata'])['config']
            expected = expected_native_config(config, rows[index], protocol, accepted, reference)
            require(manifest['native_rope'] == expected['rope'] and manifest['model'] == accepted['model'],
                    'Runtime model/rotary receipt mismatch')
            validate_output(Path(record['png']), path, rows[index], protocol, expected)
            checked.append({'row': index, 'prompt_id': rows[index]['id'], 'metadata_sha256': file_sha256(preserved),
                            'runtime_code_revision': config['code_revision'], 'independent_config_verified': True})
    require(len(checked) == 8, 'Missing native configs')
    return {'accepted': True, 'scope': 'Read-only supplemental config validation, no new generation or image judgments',
            'accepted_original_report_sha256': accepted_report_sha256, 'original_reporter_sha256': old['reporter_sha256'],
            'new_reporter_sha256': file_sha256(__file__), 'protocol_sha256': protocol_sha256,
            'archive_manifest_sha256': file_sha256(archive/'manifest.json'), 'images': 8, 'rows': checked,
            'original_report_and_archive_unchanged': True,
            'limits': ['This closes the expected-config coverage gap; earlier independent artifact audit remains separate.',
                       'Runtime model/conditioning/gain receipts are not independent model-weight or execution reconstruction.',
                       'Legacy attention config label is preserved; it is not observed backend telemetry.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--notes', type=Path, required=True)
    a = p.parse_args()
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    require(file_sha256(a.protocol) == a.protocol_sha256, 'Protocol changed')
    protocol = read_json(a.protocol)
    rows, accepted = validate_dependencies(protocol, 0, 4)
    rubric = pinned_json(protocol['rubric'])['predicate_rubric']
    presentation = pinned_json(protocol['presentation'])
    verified, raw = {}, {}
    for chunk, (start, end) in enumerate(protocol['chunks']):
        directory = a.root / f'chunk{chunk}'
        require(not (directory/'failure.json').exists(), 'Failed workflow')
        manifest, done = read_json(directory/'manifest.json'), read_json(directory/'completion.json')
        require(manifest['protocol_sha256'] == done['protocol_sha256'] == a.protocol_sha256
                and manifest['source_sha256'] == protocol['source_sha256'], 'Manifest provenance mismatch')
        require(manifest['model'] == accepted['model'] and all(manifest[k] == accepted[k]
                for k in ('torch', 'cuda', 'diffusers')), 'Model/software mismatch')
        require((done['start'], done['end']) == (start, end) and done['completed_images'] == 4
                and len(done['records']) == 4 and done['new_method'] is False, 'Incomplete chunk')
        require(read_json(directory/'progress.json') == {'status': 'completed', 'completed_images': 4},
                'Progress is not complete')
        for index, record in zip(range(start, end), done['records']):
            require(record['prompt_index'] == index, 'Row order mismatch')
            png, metadata = Path(record['png']), Path(record['metadata'])
            require(png.is_relative_to(directory) and metadata.is_relative_to(directory), 'Output escaped chunk')
            config = read_json(metadata)['config']
            reference = pinned_json(protocol['selected_prompts'][index]['reference_metadata'])['config']
            expected = expected_native_config(config, rows[index], protocol, accepted, reference)
            require(manifest['native_rope'] == expected['rope'], 'Runtime native rotary receipt mismatch')
            result = validate_output(png, metadata, rows[index], protocol, expected)
            require(all(record[k] == value for k, value in result.items()), 'Completion image receipt mismatch')
            gain = record['runtime_gain_audit']
            require(len(gain) == 28 and all(x == {'step': i, 'sigma': protocol['expected_sigmas'][i],
                'controller_float32_all_one': True, 'controller_bfloat16_all_one': True,
                'rotary_gain_all_one': True} for i, x in enumerate(gain)), 'Runtime gain receipt mismatch')
            initial = record['initial_transformer_input']
            path = Path(initial['path'])
            require(path.is_relative_to(directory) and file_sha256(path) == initial['file_sha256'],
                    'Initial input file mismatch')
            value = torch.load(path, map_location='cpu', weights_only=True)['initial_transformer_input']
            require(value.shape == (1, 4096, 64) and value.dtype == torch.bfloat16
                    and bool(torch.isfinite(value).all()) and hashlib.sha256(
                    value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == initial['tensor_sha256'],
                    'Initial input tensor mismatch')
            verified[index] = record
        raw[str(chunk)] = {'manifest': manifest, 'completion': done}
    require(set(verified) == set(range(8)), 'Missing images')
    require(not a.archive.exists(), 'Refuse to overwrite archive')
    a.archive.mkdir(parents=True)
    a.notes.mkdir(parents=True, exist_ok=True)
    retained, paths = [], {}
    for index, record in verified.items():
        selected = protocol['selected_prompts'][index]
        sources = {'native1024': Path(record['png']), 'archived4096': Path(selected['reference_png']['path']),
                   'native_metadata': Path(record['metadata']),
                   'reference_metadata': Path(selected['reference_metadata']['path']),
                   'initial_input': Path(record['initial_transformer_input']['path'])}
        for kind, source in sources.items():
            target = a.archive/kind/f'row{index}{source.suffix}'
            target.parent.mkdir(exist_ok=True)
            digest = file_sha256(source)
            shutil.copyfile(source, target)
            require(file_sha256(target) == digest, 'Archive copy mismatch')
            retained.append(dict(source=str(source), preserved=str(target), sha256=digest, bytes=target.stat().st_size))
            paths[kind, index] = target
    inputs, viewmap = [], []
    for item in presentation['mapping']:
        source = paths[item['condition'], item['row']]
        view = a.archive/'views'/f"{item['label']}.png"
        original = a.archive/'original_views'/view.name
        view.parent.mkdir(exist_ok=True)
        original.parent.mkdir(exist_ok=True)
        with Image.open(source) as image:
            image = image.convert('RGB')
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            clean = Image.new('RGB', image.size)
            clean.paste(image)
            clean.save(view)
        os.link(source, original)
        rule = rubric['rows'][item['row']]
        inputs.append({'label': item['label'], 'caption': rows[item['row']]['prompt'],
                       'primary': rule['primary'], 'secondary': rule['secondary'],
                       'auxiliary': rule.get('auxiliary', []), 'exclusions': rule['exclusions'],
                       'common_view': str(view), 'detail_view': str(original)})
        viewmap.append({**item, 'source_sha256': file_sha256(source), 'view_sha256': file_sha256(view)})
    report = {'accepted': True, 'images': 8, 'protocol_sha256': a.protocol_sha256,
              'reporter_sha256': file_sha256(__file__), 'raw': raw,
              'retained': retained, 'retained_bytes': sum(x['bytes'] for x in retained),
              'viewmap': viewmap, 'quality_scored': False,
              'limits': ['Runtime conditioning/model/gain checks are receipts; initial inputs and PNGs are saved witnesses.',
                         'Different-size noise is uncoupled. No causal resolution or method-quality conclusion.',
                         'Common-view labels precede detail inspection. Label hiding is not guaranteed resolution blinding.']}
    (a.notes/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (a.notes/'reviewer_inputs.json').write_text(json.dumps({'rules': rubric['rules'], 'labels': rubric['labels'],
        'instruction': 'Record all common-view judgments before inspecting any detail views; keep both records.',
        'images': inputs}, indent=2)+'\n')
    (a.archive/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'accepted': True, 'images': 8, 'retained_bytes': report['retained_bytes']}))


if __name__ == '__main__':
    main()
