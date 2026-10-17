"""First three held-out rows, fixed before viewing; authentic seed-0 outputs.

The asset receipt preserves source hashes, full captions, and image coordinates.
Cached display assets remain on work so this figure survives scratch purging.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as effects
from PIL import Image

import orx_figstyle
from orx_figstyle import save, use_style

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'heldout_example_assets'
ROWS = (8, 9, 10)
METHODS = ('sega_official', 'sega_ctm_v2')
CROP = (1536, 1536, 2560, 2560)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assets():
    receipt_path = ASSETS / 'provenance.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['rows'] == list(ROWS) and receipt['crop_box_xyxy'] == list(CROP)
        for row in receipt['images']:
            for key in ('full', 'crop'):
                assert digest(ASSETS / row[key]) == row[key + '_sha256']
        return receipt
    with (ROOT / '.agent/notes/main2/heldout_per_prompt.csv').open() as stream:
        scores = list(csv.DictReader(stream))
    ASSETS.mkdir(exist_ok=True)
    receipt = {'selection': 'First three held-out manifest rows, chosen before image inspection; no score selection',
               'rows': list(ROWS), 'seed': 0, 'crop_box_xyxy': list(CROP),
               'display_transform': 'Full images reduced to 768 square with Lanczos; center crops retained at native 1024 square. No enhancement.',
               'images': []}
    for index in ROWS:
        reference = None
        for method in METHODS:
            matches = [r for r in scores if int(r['idx']) == index and r['method'] == method]
            assert len(matches) == 1
            score = matches[0]
            path = Path(score['path'])
            record = json.loads(path.with_suffix('.json').read_text())
            config = record['config']
            source_hash = digest(path)
            assert source_hash == record['png_sha256'] and record['complete']
            assert config['prompt_id'] == score['prompt_id'] and config['seed'] == int(score['seed']) == 0
            assert config['method'] == ('sega_ctm' if method == 'sega_ctm_v2' else method)
            if reference is not None:
                assert all(config[k] == reference[k] for k in ('prompt_id', 'prompt', 'seed', 'derived_seed', 'steps', 'guidance'))
            else:
                reference = config
            with Image.open(path) as image:
                assert image.size == (4096, 4096) and image.mode == 'RGB'
                full, crop = f'{index}_{method}_full.png', f'{index}_{method}_center.png'
                image.resize((768, 768), Image.Resampling.LANCZOS).save(ASSETS / full)
                image.crop(CROP).save(ASSETS / crop)
            receipt['images'].append({'row': index, 'method': method, 'prompt_id': score['prompt_id'],
                                      'prompt': config['prompt'], 'source': str(path),
                                      'source_sha256': source_hash, 'full': full, 'crop': crop,
                                      'full_sha256': digest(ASSETS / full), 'crop_sha256': digest(ASSETS / crop),
                                      'sampling_config': config})
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    receipt = assets()
    use_style()
    orx_figstyle.TEXT = 6.5
    fig, axes = plt.subplots(3, 4, figsize=(6.5, 5.5))
    fig.subplots_adjust(left=.02, right=.98, top=.90, bottom=.03, hspace=.21, wspace=.045)
    headers = ('Released SEGA\nFull image', 'SEGA + constancy\nFull image',
               'Released SEGA\nCenter crop', 'SEGA + constancy\nCenter crop')
    for col, label in enumerate(headers):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, .955, label, ha='center', va='center', fontsize=8)
    labels = ('(a) Bridge, row 8', '(b) Paris, row 9', '(c) Artist corner, row 10')
    for ri, index in enumerate(ROWS):
        for col, ax in enumerate(axes[ri]):
            method = METHODS[col % 2]
            entry = next(r for r in receipt['images'] if r['row'] == index and r['method'] == method)
            key = 'full' if col < 2 else 'crop'
            with Image.open(ASSETS / entry[key]) as image:
                ax.imshow(image, interpolation='lanczos')
            ax.set_axis_off()
            if col < 2:
                box = Rectangle((288, 288), 192, 192, fill=False, edgecolor='white', linewidth=.6)
                box.set_path_effects([effects.withStroke(linewidth=1.4, foreground='black')])
                ax.add_patch(box)
        axes[ri, 0].text(0, 1.035, labels[ri], transform=axes[ri, 0].transAxes,
                        fontsize=8, weight='bold')
    stem = HERE / 'heldout_examples'
    save(fig, str(stem), formats=('pdf', 'svg'), close=False)
    fig.savefig(stem.with_suffix('.png'), dpi=180)
    plt.close(fig)
    print('Six authentic outputs, fixed rows 8-10, seed0; source and cached display hashes verified.')


if __name__ == '__main__':
    main()
