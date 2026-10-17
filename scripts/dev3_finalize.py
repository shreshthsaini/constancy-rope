"""Finalize a complete dev diagnostic and rebuild its explicitly exploratory paper table."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dev3_analysis import analyze

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
LABELS = {'sega_official': 'Released SEGA', 'sega_rms_uniform': 'RMS-matched uniform',
          'sega_rms_normalized': 'RMS-normalized allocation', 'sega_ctm_lag': 'Lag-aligned constancy'}


def main():
    queue = SCRATCH / 'taskq_dev3'
    if list((queue / 'pending').glob('*.sh')) or list((queue / 'running').glob('*.sh.*')):
        raise RuntimeError('Cannot finalize while tasks remain')
    if list((queue / 'failed').glob('*.sh')) or len(list((queue / 'done').glob('*.sh'))) != 4:
        raise RuntimeError('Require all four complete workflows with no failed tasks')
    provenance = [json.loads((SCRATCH / f'out_krea/dev3_{method}/generation_manifest.json').read_text())
                  for method in LABELS]
    if not all(p == provenance[0] for p in provenance):
        raise RuntimeError('Actual transformer provenance differs between workflows')
    out = ROOT / '.agent/notes/continuation3/dev'
    report = analyze(SCRATCH / 'eval_krea', SCRATCH / 'prompts/aesthetic4k_eval.jsonl',
                     out, reference_root='dev3_sega_official')
    (out / 'generation_provenance.json').write_text(json.dumps(provenance[0], indent=2) + '\n')
    text = [r'\paragraph{Third development diagnostic.}',
            r'The three prespecified controls and a fresh released-SEGA baseline completed on eight development prompts, seed 0. These are exploratory selection scores, not held-out efficacy evidence.',
            r'\begin{table}[ht]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
            r'\caption{Third development diagnostic. Differences are against the fresh same-hardware SEGA baseline.}',
            r'\begin{tabular}{@{}lrrrrl@{}}', r'\toprule',
            r'Variant & $\Delta$IR & $\Delta$MUSIQ & $\Delta$CLIP & $\Delta$crop MUSIQ & Gate\\',
            r'\midrule']
    for row in report['rows']:
        if row['variant'] == 'sega_official':
            continue
        values = ' & '.join(f"{row['delta_'+m]:+.4f}" for m in
                            ('imagereward', 'musiq', 'clip', 'musiq_crop1024'))
        text.append(LABELS[row['variant']] + ' & ' + values + ' & ' +
                    ('Pass' if row['eligible'] else 'Fail') + r'\\')
    text += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    text.append('PickScore differences against the same baseline are ' + '; '.join(
        LABELS[row['variant']] + f": {row['delta_pickscore']:+.4f}"
        for row in report['rows'] if row['variant'] != 'sega_official') + '.')
    if report['selected']:
        names = ', '.join(LABELS[m] for m in report['selected'])
        text += [names + ' passed the development gate. These controls remain diagnostics, not an established novel method. A separate set of 64 confirmation captions was frozen before these outcomes; its use requires a distinct method and a recent-baseline protocol. No confirmation result or efficacy claim follows from this table.']
    else:
        text += ['None passed the frozen development gate, so no confirmation generation is justified by this diagnostic. The negative outcome is retained with all three variants.']
    target = ROOT / 'paper/sec/dev3_results.tex'
    target.write_text('\n'.join(text) + '\n')
    import wandb
    mode = os.environ.get('WANDB_MODE', 'online' if os.environ.get('WANDB_API_KEY') else 'offline')
    run = wandb.init(project='constancy-rope', group='dev3-diagnostics', name='dev3-final-selection',
                     dir=str(SCRATCH / 'wandb'), mode=mode, job_type='analysis',
                     config={'purpose': report['purpose'], 'reference': report['reference_root']})
    for row in report['rows']:
        run.log(row)
    run.finish()
    build = SCRATCH / 'paper_build/dev3_complete'
    build.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RAYON_NUM_THREADS='1')
    subprocess.run(['/home1/09032/saini_2/.local/bin/tectonic', '-C', '--keep-logs',
                    '--outdir', str(build), str(ROOT / 'paper/main.tex')], cwd=ROOT, env=env, check=True)
    pdf = ROOT / 'paper/constancy_rope_working_2026-10-17.pdf'
    shutil.copy2(build / 'main.pdf', pdf)
    receipt = {'status': 'development diagnostic complete', 'selected': report['selected'],
               'confirmation_run': False, 'paper': str(pdf),
               'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    state_path = ROOT / '.agent/STATE.md'
    state = state_path.read_text()
    next_action = ('Review the causal diagnostic and establish a distinct method against recent prior art before confirmation.'
                   if report['selected'] else
                   'Third diagnostic failed the development gate; preserve the result and develop a distinct mechanism against recent prior art.')
    state = '\n'.join('NEXT: ' + next_action if line.startswith('NEXT:') else line
                      for line in state.splitlines()) + '\n'
    state = state.replace('CR14 job1076857 is PENDING Priority; no estimated start.',
                          'CR14 development workflows and analysis COMPLETE; see notes/continuation3/dev/completion.json.')
    state = state.replace('no new GPU results yet.',
                          'development results archived in notes/continuation3/dev/.')
    state_path.write_text(state)
    handover_path = ROOT / 'HANDOVER.md'
    handover = handover_path.read_text().replace(
        'currently PENDING Priority on two GH200 nodes.',
        'development workflows complete on two GH200 nodes; see `.agent/notes/continuation3/dev/completion.json`.')
    handover = handover.replace(
        '- Current step: monitor job1076857, verify telemetry and source/protocol invariants, then\n'
        '  run `scripts/dev3_analysis.py` with `--reference-root dev3_sega_official`. No GPU outcome yet.',
        '- Current step: ' + next_action + '\n'
        '  Development scores and selection are archived in `.agent/notes/continuation3/dev/`.')
    handover_path.write_text(handover)
    paths = ['paper/sec/dev3_results.tex', 'paper/constancy_rope_working_2026-10-17.pdf',
             '.agent/notes/continuation3/dev', '.agent/STATE.md', 'HANDOVER.md']
    subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Record completed third development diagnostic and refresh paper',
                    '--', *paths], cwd=ROOT, check=True)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
