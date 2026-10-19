"""Matched-guidance checkpoint figure, gated on the complete P2 descriptive report.

Uses development row 0, all four checkpoint/guidance cells, and fixed native crops.
No prompt search, placeholder figure, model loading, or generation occurs here.
The historical unmatched-guidance f5_checkpoint.py remains separately reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
from PIL import Image

import crope_style as cs
from crope.utils import file_sha256, runtime_environment
from scripts import strengthen_diagnostic_report as diagnostic

BOXES = {'A':(1952,928,2144,1120), 'B':(1952,2976,2144,3168)}
INK = {'dev':cs.INK, 'krea':cs.COBALT}
CELLS = ((3.5, diagnostic.CELLS[0], diagnostic.CELLS[2]),
         (4.5, diagnostic.CELLS[1], diagnostic.CELLS[3]))


def checked_report(path):
    report = json.loads(Path(path).read_text())
    diagnostic.verify_report_images(report)
    if report['block'] != 'ckpt_matrix':
        raise ValueError('Matched checkpoint figure requires the complete P2 report')
    if report['reporter_sha256'] != file_sha256(diagnostic.__file__):
        raise ValueError('Reporter source differs from the completed report')
    for source, digest in report['input_sha256'].items():
        if file_sha256(source) != digest:
            raise ValueError(f'Report input changed: {source}')
    manifest_path = diagnostic.NOTES/'ckpt_matrix_queue.json'
    if str(manifest_path) not in report['input_sha256']:
        raise ValueError('Report lacks the frozen checkpoint-matrix manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest['prompts'] not in report['input_sha256']:
        raise ValueError('Report lacks its fixed prompt manifest')
    fixed_ids = [row['id'] for row in diagnostic.read_prompts(manifest['prompts'])[:8]]
    if report['prompt_ids'] != fixed_ids:
        raise ValueError('Report prompt order differs from the fixed development rows')
    if report.get('lattice_definition') != diagnostic.LATTICE_DEFINITION:
        raise ValueError('Whitened-lattice definition differs')
    expected = {(method,prompt) for method in diagnostic.CELLS for prompt in report['prompt_ids']}
    spectra = report.get('spectra', [])
    if len(spectra) != 32 or {(s['method'],s['prompt_id']) for s in spectra} != expected:
        raise ValueError('All 32 full-frame spectra are required')
    reference = np.sqrt(np.geomspace(2/4096,.5,49)[:-1]*np.geomspace(2/4096,.5,49)[1:])
    for s in spectra:
        f, power = np.asarray(s['frequency']), np.asarray(s['whitened_power'])
        if f.shape != (48,) or not np.array_equal(f, reference):
            raise ValueError('Spectral frequency bins differ from the fixed 4K definition')
        if power.shape != (48,) or not np.isfinite(power).all() or (power <= 0).any():
            raise ValueError('All plotted spectral powers must be finite and positive')
    # Re-derive the caption summaries from the saved full spectra before displaying them.
    spectral = {(s['method'],s['prompt_id']):s for s in spectra}
    recomputed = []
    for guidance,dev,krea in CELLS:
        values = [diagnostic.peak_statistic(reference, spectral[(dev,p)]['whitened_power'],
                                            spectral[(krea,p)]['whitened_power'])['ratio']
                  for p in report['prompt_ids']]
        recomputed.append({'guidance':guidance, 'statistic':'dev/Krea power at dev-selected peak',
                           **diagnostic.bootstrap(values, geometric=True)})
    if report['lattice_summary'] != recomputed:
        raise ValueError('Reported eight-prompt lattice summaries differ from saved spectra')
    return report


def caption(report):
    estimates = []
    for summary in report['lattice_summary']:
        estimates.append(f"guidance {summary['guidance']:g}: {summary['mean']:.2f} "
                         f"[95% interval {summary['ci_low']:.2f}, {summary['ci_high']:.2f}]")
    return ('Checkpoint and guidance diagnostic using official DyPE code. Rows use guidance 3.5 and 4.5; '
            'columns compare FLUX.1-dev and FLUX.1-Krea-dev with the same base pipeline, 28 steps and seed 42. '
            'The displayed prompt is development row 0, fixed before generation. Full frames are reduced '
            'with Lanczos; A and B are untouched 192 by 192 pixel crops centered at (2048, 1024) and '
            '(2048, 3072), with identical coordinates in every image. Right panels show full-frame '
            'Hann-windowed radial luminance power multiplied by frequency squared for the displayed pair, '
            'on shared axes. Across all eight fixed development prompts, the geometric mean dev/Krea '
            'whitened-power ratio at the dev-selected peak over periods 8 to 64 pixels is '
            + '; '.join(estimates) + '. Intervals use 2,000 paired-prompt bootstrap resamples. '
            'These summaries are descriptive: selecting the peak from the dev image can favor dev in the ratio; '
            'they do not estimate a general failure rate or establish a confirmatory result.')


def add_axis(fig, x, top, width, height):
    w,h = fig.get_size_inches()
    return fig.add_axes([x/w, 1-(top+height)/h, width/w, height/h])


def photograph(ax, array):
    ax.imshow(array, interpolation='none')
    ax.set_axis_off()


def make_figure(report_path, out):
    report = checked_report(report_path)
    out = Path(out)
    if any(out.with_suffix(suffix).exists() for suffix in ('.pdf','.svg','.json','.caption.txt')):
        raise FileExistsError('Preserve existing figure artifacts by choosing a new output stem')
    prompt_id = report['prompt_ids'][0]
    images = {(s['method'],s['prompt_id']):s for s in report['images']}
    spectra = {(s['method'],s['prompt_id']):s for s in report['spectra']}
    displayed = [images[(method,prompt_id)] for _,dev,krea in CELLS for method in (dev,krea)]
    powers = np.concatenate([10*np.log10(spectra[(s['method'],prompt_id)]['whitened_power']) for s in displayed])
    limits = (float(5*np.floor(powers.min()/5)-1), float(5*np.ceil(powers.max()/5)+1))
    cs.style()
    width, height = cs.TEXT, 2.67
    frame, gap, crop = .92, .03, .445
    x_cell = {'dev':.55, 'krea':2.045}
    cell_width = frame+gap+crop
    spec_x, spec_width = 3.94, width-3.94-.035
    fig = plt.figure(figsize=(width,height))
    fig.text(.55/width, 1-.12/height, '(a) Fixed development prompt 0', fontsize=7.5, color=cs.INK)
    fig.text(spec_x/width, 1-.12/height, '(b) Whitened spectra', fontsize=7.5, color=cs.INK)
    for checkpoint,name in (('dev','FLUX.1-dev'),('krea','FLUX.1-Krea-dev')):
        fig.text((x_cell[checkpoint]+cell_width/2)/width, 1-.29/height, name,
                 fontsize=7.2, weight='bold', color=INK[checkpoint], ha='center')
    pixel_metadata = []
    for row,(guidance,dev,krea) in enumerate(CELLS):
        top = .36+row*1.075
        fig.text(0,1-(top+.31)/height,'Guidance',fontsize=6.7,color=cs.INK2)
        fig.text(0,1-(top+.46)/height,f'{guidance:g}',fontsize=8.5,weight='bold',color=cs.INK)
        for checkpoint,method in (('dev',dev),('krea',krea)):
            witness = images[(method,prompt_id)]
            with Image.open(witness['path']) as source:
                original = source.convert('RGB')
                thumbnail = np.asarray(original.resize((768,768), Image.Resampling.LANCZOS))
                cropped = {tag:np.asarray(original.crop(box)) for tag,box in BOXES.items()}
            ax = add_axis(fig,x_cell[checkpoint],top,frame,frame)
            photograph(ax,thumbnail)
            for index,(tag,box) in enumerate(BOXES.items()):
                scale = 768/4096
                rect = Rectangle((box[0]*scale,box[1]*scale),192*scale,192*scale,
                                 fill=False,color='white',linewidth=.6)
                rect.set_path_effects([pe.withStroke(linewidth=1.6,foreground=cs.INK)])
                ax.add_patch(rect)
                text = ax.text(box[2]*scale+9,box[1]*scale+192*scale/2,tag,color='white',fontsize=6)
                text.set_path_effects([pe.withStroke(linewidth=1.5,foreground=cs.INK)])
                crop_axis = add_axis(fig,x_cell[checkpoint]+frame+gap,top+index*(crop+gap),crop,crop)
                photograph(crop_axis,cropped[tag])
                crop_axis.text(.03,.96,tag,transform=crop_axis.transAxes,color='white',fontsize=6,
                               va='top',path_effects=[pe.withStroke(linewidth=1.5,foreground=cs.INK)])
                pixel_metadata.append({'method':method,'prompt_id':prompt_id,'crop':tag,'box':list(box),
                    'native_rgb_sha256':hashlib.sha256(cropped[tag].tobytes()).hexdigest()})
        ax = add_axis(fig,spec_x,top+.025,spec_width,frame-.07)
        for checkpoint,method in (('dev',dev),('krea',krea)):
            values = spectra[(method,prompt_id)]
            ax.plot(values['frequency'],10*np.log10(values['whitened_power']),
                    color=INK[checkpoint],linewidth=1.1,label='dev' if checkpoint=='dev' else 'Krea')
        ax.set_xscale('log')
        ax.set_xlim(.0005,.5)
        ax.set_ylim(*limits)
        ax.xaxis.set_major_locator(FixedLocator([.001,.01,.1]))
        ax.set_xticklabels(['0.001','0.01','0.1'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=6,length=2,pad=2)
        ax.set_ylabel('Power (dB)',fontsize=6.5,labelpad=2)
        ax.spines[['top','right']].set_visible(False)
        if row == 0:
            ax.legend(loc='lower left',fontsize=6,frameon=False,handlelength=1.4,borderpad=0)
        else:
            ax.set_xlabel('Frequency (cycles/pixel)',fontsize=6.5,labelpad=3)
    out.parent.mkdir(parents=True,exist_ok=True)
    cs.finish(fig,out,photo=False,formats=('pdf','svg'))
    plt.close(fig)
    metadata = {'schema':'matched-checkpoint-figure-v1','report':str(Path(report_path).resolve()),
                'report_sha256':file_sha256(report_path),'source_sha256':file_sha256(__file__),
                'prompt_row':0,'prompt_id':prompt_id,'image_witnesses':displayed,
                'crop_rule':'Fixed 192 pixel windows at frame quarter and three-quarter heights on vertical midline',
                'native_crops':pixel_metadata,'spectrum_y_limits_db':list(limits),
                'lattice_definition':report['lattice_definition'],'lattice_summary':report['lattice_summary'],
                'lattice_per_prompt':report['lattice_per_prompt'],'caption':caption(report),
                'native_crop_encoding':'Lossless embedded raster; no resampling or JPEG postprocessing'}
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    out.with_suffix('.caption.txt').write_text(caption(report)+'\n')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,default=HERE/'f5_checkpoint_matched')
    args = p.parse_args()
    runtime_environment()
    try:
        make_figure(args.report,args.out)
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        print(f'FIGURE NOT WRITTEN: {error}',file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
