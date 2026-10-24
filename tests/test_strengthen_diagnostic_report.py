"""Small CPU contracts for descriptive diagnostics; no model or result figures."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import strengthen_diagnostic_report as report


def design(block='ckpt_matrix'):
    prompts = [{'id': f'p{i}', 'prompt': f'caption {i}'} for i in range(9)]
    methods = report.CELLS if block == 'ckpt_matrix' else ('sega_default_shift',)
    seed = 42 if block == 'ckpt_matrix' else 0
    samples = [{'method': method, 'prompt_id': row['id'], 'seed': seed}
               for method in methods for row in prompts[:8]]
    return {'block': block}, prompts, samples


@pytest.mark.parametrize('block', ['ckpt_matrix', 'default_shift'])
def test_complete_fixed_design(block):
    manifest, prompts, samples = design(block)
    report.validate_design(manifest, samples, prompts)
    with pytest.raises(ValueError, match='requires exactly'):
        report.validate_design(manifest, samples[:-1], prompts)
    with pytest.raises(ValueError, match='requires exactly'):
        report.validate_design(manifest, samples + samples[:1], prompts)


@pytest.mark.parametrize('key,value', [('prompt_id','p8'), ('seed',1), ('method','other')])
def test_fixed_design_rejects_substitution(key, value):
    manifest, prompts, samples = design()
    samples[0][key] = value
    with pytest.raises(ValueError, match='requires exactly'):
        report.validate_design(manifest, samples, prompts)


def test_paired_bootstrap_and_geometric_statistic():
    assert report.bootstrap(np.full(8, 2.)) == {'n':8, 'mean':2., 'ci_low':2., 'ci_high':2.}
    values = np.arange(1., 9.)
    got = report.bootstrap(values)
    assert got == report.bootstrap(values)
    assert got['ci_low'] < got['mean'] == 4.5 < got['ci_high']
    ratios = np.array([1., 2.] * 4)
    assert report.bootstrap(ratios, geometric=True)['mean'] == pytest.approx(np.sqrt(2))
    for invalid in (np.ones(7), np.r_[np.ones(7), np.nan]):
        with pytest.raises(ValueError):
            report.bootstrap(invalid)
    with pytest.raises(ValueError):
        report.bootstrap(np.arange(8.), geometric=True)


def test_whitening_matches_independent_radial_bin_average():
    # Direct masked means avoid copying the implementation's bincount calculation.
    n = 128
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    base = np.uint8(np.clip(127 + 60*np.cos(2*np.pi*x/16) + 30*np.sin(2*np.pi*y/32), 0, 255))
    rgb = np.stack([base, np.roll(base, 3, axis=0), base], axis=-1)
    f, power = report.whitened_array(rgb)
    luminance = (rgb/255.) @ np.array([.2126,.7152,.0722])
    centered = luminance-luminance.mean()
    window = np.outer(np.hanning(n), np.hanning(n))
    periodogram = np.abs(np.fft.fft2(centered*window))**2 / n**2
    frequency = np.fft.fftfreq(n)
    radius = np.sqrt(frequency[:,None]**2+frequency[None,:]**2)
    edges = np.geomspace(2/n, .5, 49)
    expected = np.array([periodogram[(radius >= lo) & (radius < hi)].mean()
                         if ((radius >= lo) & (radius < hi)).any() else 0
                         for lo,hi in zip(edges[:-1],edges[1:])]) * f**2
    np.testing.assert_allclose(power, expected, rtol=2e-13, atol=1e-20)


def test_peak_band_is_strict_and_selected_from_dev():
    f = np.array([1/128, 1/64, 1/32, 1/16, 1/8, 1/4])
    dev = np.array([100,100,4,4,100,100.])
    krea = np.array([1,1,2,10,1,1.])
    got = report.peak_statistic(f, dev, krea)
    assert got['bin'] == 2 and got['ratio'] == 2 and got['period_pixels'] == 32
    with pytest.raises(ValueError):
        report.peak_statistic(f, dev, krea*0)


def scored_fixture(tmp_path):
    sample = {'method':'m', 'prompt_id':'p0', 'seed':42, 'res':'4096x4096',
              'path':str(tmp_path/'image.png')}
    a = {**sample, **{key:float(i) for i,key in enumerate(report.METRICS) if key != 'imagereward'}}
    b = {key:sample[key] for key in ('method','prompt_id','res')}
    b.update(png=sample['path'], imagereward=.5)
    first, second = tmp_path/'metrics.csv', tmp_path/'reward.csv'
    return sample, a, b, first, second


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'identity', 'seed', 'nonfinite'])
def test_score_join_rejects_invalid_evidence(tmp_path, failure):
    sample, a, b, first, second = scored_fixture(tmp_path)
    if failure == 'missing':
        b['png'] = str(tmp_path/'other.png')
    if failure == 'identity':
        b['method'] = 'wrong'
    if failure == 'seed':
        a['seed'] = 0
    if failure == 'nonfinite':
        a['musiq'] = np.nan
    pd.DataFrame([a,a] if failure == 'duplicate' else [a]).to_csv(first,index=False)
    pd.DataFrame([b]).to_csv(second,index=False)
    with pytest.raises(ValueError):
        report.load_metrics([sample], first, second)


def test_score_join_and_paired_contrast(tmp_path):
    sample, a, b, first, second = scored_fixture(tmp_path)
    pd.DataFrame([a]).to_csv(first,index=False)
    pd.DataFrame([b]).to_csv(second,index=False)
    got = report.load_metrics([sample], first, second)
    assert got.iloc[0].imagereward == .5
    rows = []
    for method, offset in [('sega_default_shift',3.), ('sega_official',1.)]:
        for i in range(8):
            rows.append({'method':method, 'prompt_id':f'p{i}',
                         **{metric:i+offset for metric in report.METRICS}})
    means, paired = report.summaries(pd.DataFrame(rows), 'default_shift', [f'p{i}' for i in range(8)])
    assert len(means) == 20 and len(paired) == 10
    assert all(row['mean'] == row['ci_low'] == row['ci_high'] == 2 for row in paired)


def report_fixture(tmp_path):
    manifest, prompts, samples = design()
    for i,sample in enumerate(samples):
        png = tmp_path/f'{i}.png'
        side = png.with_suffix('.json')
        png.write_bytes(f'evidence {i}'.encode())
        side.write_text('{}')
        sample.update(path=str(png), sidecar=str(side), png_sha256=report.file_sha256(png),
                      sidecar_sha256=report.file_sha256(side))
    return {'schema':'strengthen-descriptive-diagnostics-v1', 'block':manifest['block'], 'complete':True,
            'prompt_rows':list(range(8)), 'prompt_ids':[p['id'] for p in prompts[:8]], 'images':samples}


def test_figure_gate_rejects_incomplete_duplicate_or_changed_evidence(tmp_path):
    complete = report_fixture(tmp_path)
    report.verify_report_images(complete)
    incomplete, duplicate = deepcopy(complete), deepcopy(complete)
    incomplete['images'].pop()
    duplicate['images'][-1] = deepcopy(duplicate['images'][0])
    for broken in (incomplete, duplicate):
        with pytest.raises(ValueError, match='incomplete image set'):
            report.verify_report_images(broken)
    complete['images'][0]['png_sha256'] = 'changed'
    with pytest.raises(ValueError, match='evidence changed'):
        report.verify_report_images(complete)


def test_incomplete_block_writes_no_report(tmp_path, monkeypatch):
    from scripts import strengthen_evaluation as gate
    manifest, prompts, samples = design()
    manifest['prompts'] = str(tmp_path/'prompts.jsonl')
    (tmp_path/'prompts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in prompts))
    (tmp_path/'ckpt_matrix_queue.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(report, 'NOTES', tmp_path)
    monkeypatch.setattr(gate, 'expected_samples', lambda _: samples)
    monkeypatch.setattr(gate, 'completion_status', lambda *a: {
        'complete':False, 'missing_image_pairs':['missing'], 'unfinished_tasks':['queued']})
    output = tmp_path/'result'
    with pytest.raises(ValueError, match='incomplete'):
        report.build_report('ckpt_matrix', output)
    assert not output.exists()


def figure_fixture(tmp_path, monkeypatch):
    from paper.figs import f5_checkpoint_matched as figure
    complete = report_fixture(tmp_path)
    prompt_path = tmp_path/'prompts.jsonl'
    prompt_path.write_text(''.join(json.dumps({'id':p,'prompt':'test fixture'})+'\n'
                                  for p in complete['prompt_ids']))
    manifest = tmp_path/'ckpt_matrix_queue.json'
    manifest.write_text(json.dumps({'prompts':str(prompt_path)}))
    monkeypatch.setattr(report,'NOTES',tmp_path)
    complete.update(reporter_sha256=report.file_sha256(report.__file__),
                    input_sha256={str(p):report.file_sha256(p) for p in (manifest,prompt_path)},
                    lattice_definition=deepcopy(report.LATTICE_DEFINITION))
    edges = np.geomspace(2/4096,.5,49)
    f = np.sqrt(edges[:-1]*edges[1:])
    complete['spectra'] = [{'method':m,'prompt_id':p,'frequency':f.tolist(),
                            'whitened_power':np.full(48,2. if '_dev_' in m else 1.).tolist()}
                           for m in report.CELLS for p in complete['prompt_ids']]
    complete['lattice_summary'] = [{'guidance':g,'statistic':'dev/Krea power at dev-selected peak',
                                   **report.bootstrap(np.full(8,2.),geometric=True)} for g in (3.5,4.5)]
    path = tmp_path/'report.json'
    path.write_text(json.dumps(complete))
    return figure, complete, path


def test_figure_gate_verifies_complete_spectra_and_caption_summary(tmp_path, monkeypatch):
    figure, complete, path = figure_fixture(tmp_path, monkeypatch)
    figure.checked_report(path)
    assert figure.BOXES == {'A':(1952,928,2144,1120), 'B':(1952,2976,2144,3168)}
    assert 'development row 0' in figure.caption(complete)
    assert 'selecting the peak from the dev image' in figure.caption(complete)
    complete['lattice_summary'][0]['mean'] = 9
    path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match='summaries differ'):
        figure.checked_report(path)


def test_figure_gate_rejects_spectral_subset_before_creating_artifacts(tmp_path, monkeypatch):
    figure, complete, path = figure_fixture(tmp_path, monkeypatch)
    complete['spectra'].pop()
    path.write_text(json.dumps(complete))
    out = tmp_path/'figures'/'checkpoint'
    with pytest.raises(ValueError, match='All 32'):
        figure.make_figure(path,out)
    assert not out.parent.exists()


def test_figure_gate_rejects_reordered_prompt_selection(tmp_path, monkeypatch):
    figure, complete, path = figure_fixture(tmp_path, monkeypatch)
    complete['prompt_ids'].reverse()
    path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match='prompt order differs'):
        figure.checked_report(path)
