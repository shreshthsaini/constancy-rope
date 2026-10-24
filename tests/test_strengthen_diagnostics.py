"""Contract checks for the official-code checkpoint diagnostic; no model loads."""

import json
from types import SimpleNamespace

import pytest

from scripts import strengthen_diagnostics as diagnostic


def args(tmp_path):
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text(json.dumps({'id': 'a', 'prompt': 'A green tree'}) + '\n')
    return SimpleNamespace(checkpoint='dev', guidance=3.5, seed=42, steps=28, prompts=prompts)


def test_matrix_has_unique_evaluator_identities():
    names = {diagnostic.method_name(checkpoint, guidance)
             for checkpoint in ('dev', 'krea') for guidance in (3.5, 4.5)}
    assert len(names) == 4


def test_checkpoint_control_changes_only_transformer_and_identity(tmp_path):
    settings = args(tmp_path)
    row = {'id': 'a', 'prompt': 'A green tree'}
    dev = diagnostic.build_config(settings, row, {'source': 'fixed'}, {'torch': 'fixed'})
    settings.checkpoint = 'krea'
    krea = diagnostic.build_config(settings, row, {'source': 'fixed'}, {'torch': 'fixed'})
    differences = {key for key in dev if dev[key] != krea[key]}
    assert differences == {'method', 'checkpoint', 'transformer_model', 'transformer_revision'}
    assert dev['base_revision'] == krea['base_revision'] == diagnostic.DEV_REVISION


def test_guidance_control_changes_only_guidance_and_identity(tmp_path):
    settings = args(tmp_path)
    row = {'id': 'a', 'prompt': 'A green tree'}
    low = diagnostic.build_config(settings, row, {}, {})
    settings.guidance = 4.5
    high = diagnostic.build_config(settings, row, {}, {})
    assert {key for key in low if low[key] != high[key]} == {'method', 'guidance'}


def test_seed_matches_old_official_diagnostic_for_every_prompt(tmp_path):
    settings = args(tmp_path)
    first = diagnostic.build_config(settings, {'id': 'a', 'prompt': 'A'}, {}, {})
    second = diagnostic.build_config(settings, {'id': 'b', 'prompt': 'B'}, {}, {})
    assert first['resolved_seed'] == second['resolved_seed'] == 42
    assert 'device=cuda' in first['generator']


@pytest.mark.parametrize('start,end,seed,steps', [(0,9,42,28), (8,9,42,28), (0,8,0,28), (0,8,42,4)])
def test_protocol_violations_rejected_before_loading(start, end, seed, steps):
    with pytest.raises(ValueError):
        diagnostic.validate_inputs(SimpleNamespace(start=start, end=end, seed=seed, steps=steps))


def test_resume_preserves_incomplete_or_mismatched_files(tmp_path):
    png, metadata = tmp_path / 'image.png', tmp_path / 'image.json'
    assert not diagnostic.completed_sample(png, metadata, {})
    png.write_bytes(b'preserve me')
    with pytest.raises(ValueError, match='Incomplete'):
        diagnostic.completed_sample(png, metadata, {})
    metadata.write_text(json.dumps({'complete': True, 'config': {'seed': 0}}))
    with pytest.raises(ValueError, match='configuration differs'):
        diagnostic.completed_sample(png, metadata, {'seed': 42})
    assert png.read_bytes() == b'preserve me'


def test_resume_validates_png_hash_before_opening(tmp_path):
    png, metadata = tmp_path / 'image.png', tmp_path / 'image.json'
    png.write_bytes(b'corrupted')
    metadata.write_text(json.dumps({'complete': True, 'config': {}, 'png_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='hash differs'):
        diagnostic.completed_sample(png, metadata, {})


def test_unmodified_official_source_required(tmp_path, monkeypatch):
    answers = iter([diagnostic.OFFICIAL_COMMIT, ' M flux/pipeline_flux.py'])
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', lambda *a, **k: next(answers))
    with pytest.raises(ValueError, match='worktree is modified'):
        diagnostic.source_provenance(tmp_path)


def test_wrong_official_commit_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', lambda *a, **k: 'wrongcommit')
    with pytest.raises(ValueError, match='commit differs'):
        diagnostic.source_provenance(tmp_path)
