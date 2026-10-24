"""CPU checks for the Qwen control contrasts and smoke gate."""

import json
from types import SimpleNamespace

import pytest

from scripts import strengthen_qwen as qwen


def test_official_import_and_cpu_parity():
    result = qwen.cpu_checks()
    assert result['passed']
    assert result['native_rotary_max_abs_error'] == 0
    assert result['tiny_native_transformer_max_abs_error'] == 0
    assert result['uniform_energy_independent']
    assert result['native_after_other_arms_exact']


def test_config_preserves_evaluator_prompt_and_records_effective_prompt():
    row={'id':'caption1','prompt':'A green tree'}
    configs=[qwen.fixed_config(arm,row,{}) for arm in qwen.ARMS]
    for config in configs:
        assert config['prompt'] == row['prompt']
        assert config['effective_prompt'] == row['prompt'] + qwen.SUFFIX
        assert config['negative_prompt'] == ''
        assert config['resolved_seed'] == 0
        assert config['steps'] == 40
        assert config['mu'] == .9
        assert config['shift_terminal'] == .02
        assert config['frame_qk_amplitude'] == 1
    assert configs[0]['spatial_ntk_factor'] == 1
    assert configs[1]['spatial_ntk_factor'] == configs[2]['spatial_ntk_factor'] > 1
    assert configs[1]['mscale_alpha'] == .15
    assert configs[2]['mscale_alpha'] == 0


def test_smoke_has_two_production_resolution_steps_without_quality_claim():
    config=qwen.fixed_config('qwen_sega',{'id':'a','prompt':'A'},{},smoke=True)
    assert config['steps'] == 2
    assert config['res'] == '4096x4096'
    assert config['smoke'] is True


def test_arm_override_restores_after_exception():
    def forward(**kwargs):
        return kwargs
    rope=SimpleNamespace(forward=forward,mscale_alpha=.15)
    transformer=SimpleNamespace(pos_embed=rope)
    with pytest.raises(RuntimeError):
        with qwen.arm_control(transformer,'qwen_base'):
            seen=rope.forward(ntk_factor_h=17,ntk_factor_w=17,mscale_spread=.8)
            assert seen['ntk_factor_h'] == seen['ntk_factor_w'] == 1
            assert seen['mscale_spread'] is None
            raise RuntimeError('test interruption')
    assert rope.forward is forward
    assert rope.mscale_alpha == .15
    with qwen.arm_control(transformer,'qwen_temp_only'):
        assert rope.mscale_alpha == 0
    assert rope.mscale_alpha == .15


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        with qwen.arm_control(None,'unknown'):
            pass


def test_existing_corrupted_or_incomplete_output_is_preserved(tmp_path):
    png=tmp_path/'a.png'
    sidecar=tmp_path/'a.json'
    png.write_bytes(b'preserve me')
    with pytest.raises(ValueError,match='incomplete'):
        qwen.verify_existing(png,sidecar,{})
    sidecar.write_text(json.dumps({'complete':True,'config':{},'png_sha256':'wrong'}))
    with pytest.raises(ValueError,match='hash differs'):
        qwen.verify_existing(png,sidecar,{})
    assert png.read_bytes() == b'preserve me'
