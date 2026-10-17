"""Generation preflight must fail before loading a model when selection is absent."""
import json

import pytest

from scripts import confirmation_generate as generate
from scripts.confirmation_analysis import METRICS, VARIANTS, decision, sha256


@pytest.mark.parametrize('start,end', [(-1, 8), (0, 9), (0, 65), (8, 8), (60, 68)])
def test_invalid_chunk_is_rejected_before_reading_selection(start, end):
    with pytest.raises(ValueError, match='at most eight'):
        generate.validated_chunk('sega_official', start, end)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    notes = tmp_path / 'notes'
    (notes / 'dev').mkdir(parents=True)
    monkeypatch.setattr(generate, 'ROOT', tmp_path)
    monkeypatch.setattr(generate, 'NOTES', notes)
    means = {m: {k: 1. for k in METRICS} for m in ('sega_official', *VARIANTS)}
    means['sega_rms_uniform'].update(imagereward=2., musiq=2.)
    rows, selected = decision(means)
    (notes / 'dev/selection.json').write_text(json.dumps({
        'reference_root': 'dev3_sega_official', 'rows': rows, 'selected': selected}))
    prompts = notes / 'confirmation64_prompts.jsonl'
    prompts.write_text(''.join(json.dumps({'id': str(i), 'prompt': f'caption {i}'}) + '\n'
                               for i in range(64)))
    monkeypatch.setattr(generate, 'PROMPT_SHA256', sha256(prompts))
    (tmp_path / 'crope').mkdir()
    controller = tmp_path / 'crope/gain.py'
    controller.write_text('fixture source')
    (notes / 'frozen_manifest.json').write_text(json.dumps({'source_sha256': {str(controller): sha256(controller)}}))
    transformer = tmp_path / 'pinned-snapshot'
    (transformer / 'transformer').mkdir(parents=True)
    config = transformer / 'transformer/config.json'
    config.write_text('{}')
    provenance = {'transformer_snapshot': str(transformer), 'transformer_config_sha256': sha256(config)}
    (notes / 'dev/generation_provenance.json').write_text(json.dumps(provenance))
    monkeypatch.setenv('CROPE_TRANSFORMER', str(transformer))
    return notes, controller


def test_valid_chunk_keeps_exact_frozen_order(ready):
    prompts, _ = generate.validated_chunk('sega_rms_uniform', 8, 16)
    assert [r['id'] for r in prompts] == [str(i) for i in range(8, 16)]


def test_unselected_method_cannot_generate(ready):
    with pytest.raises(ValueError, match='unselected'):
        generate.validated_chunk('sega_ctm_lag', 0, 8)


def test_controller_change_cannot_generate(ready):
    ready[1].write_text('changed after development')
    with pytest.raises(ValueError, match='controller or profile changed'):
        generate.validated_chunk('sega_official', 0, 8)
