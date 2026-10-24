#!/usr/bin/env python3
"""Stage Qwen smoke or smoke-gated transfer tasks for the existing worker spool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from crope.utils import file_sha256,read_prompts
from scripts.strengthen_qwen import ARMS,MODEL_REVISION,SCRATCH,source_provenance

NOTES=ROOT/'.agent/notes/strengthen'
QUEUE=SCRATCH/'taskq'
PYTHON='/work/09032/saini_2/vista/library/uv-envs/crope/bin/python'


def prepare(block):
    manifest=NOTES/(block+'_queue.json')
    if manifest.exists():
        raise FileExistsError(manifest)
    for previous in ('fresh','seeds'):
        if not (NOTES/(previous+'_published.json')).is_file():
            raise ValueError('P4/P5 must be published before Qwen preparation')
    checks=json.loads((NOTES/'qwen_cpu_checks_final.json').read_text())
    provenance=source_provenance()
    if not checks['checks']['passed'] or checks['provenance'] != provenance:
        raise ValueError('Matching successful CPU parity checks required')
    if block=='qwen_smoke':
        prompts=SCRATCH/'prompts/aesthetic4k_eval.jsonl'
        planned=[(None,0,1)]
    elif block=='qwen':
        receipt=json.loads((SCRATCH/'diag/qwen_smoke/passed.json').read_text())
        if not receipt.get('passed') or receipt.get('provenance') != provenance or receipt.get('cpu_offload'):
            raise ValueError('Matching successful GPU smoke with whole-pipeline GPU placement required')
        if receipt.get('model_revision') != MODEL_REVISION:
            raise ValueError('Wrong GPU smoke checkpoint')
        prompts=ROOT/'prompts/fresh150.jsonl'
        if len(read_prompts(prompts)) != 150:
            raise ValueError('Require the frozen150 fresh prompts')
        planned=[(arm,start,min(start+7,150)) for start in range(0,150,7) for arm in ARMS]
    else:
        raise ValueError(block)
    templates=list((QUEUE/'done').glob('50_c00000_main_sega_official_*.sh'))
    if len(templates)!=1:
        raise ValueError('Expected one archived generation environment')
    template=templates[0]
    env=[line for line in template.read_text().splitlines() if not line.startswith('[[')
         and 'scripts/sample.py' not in line and 'fingerprint:' not in line
         and not line.startswith('export CROPE_ROPE=') and not line.startswith('export CROPE_TRANSFORMER=')]
    env+=['unset CROPE_ROPE CROPE_TRANSFORMER']
    sources=[Path(p) for p in provenance['source_sha256']]
    sources += [Path(__file__),prompts,template,NOTES/'qwen_cpu_checks_final.json',NOTES/'qwen_hub_manifest.json']
    checksum=NOTES/(block+'_generation.sha256')
    with checksum.open('x') as stream:
        stream.write(''.join(f'{file_sha256(p)}  {p}\n' for p in sources))
    staging=QUEUE/'staging/strengthen'/block
    staging.mkdir(parents=True,exist_ok=True)
    rows=read_prompts(prompts)
    tasks=[]
    for arm,start,end in planned:
        if block=='qwen_smoke':
            name='70_qwen_smoke.sh'
            out=SCRATCH/'diag/qwen_smoke'
            command=[PYTHON,'scripts/strengthen_qwen.py','smoke','--prompts',str(prompts),
                     '--start','0','--end','1','--out',str(out)]
            method='qwen_smoke_three_arms'
            n=3
        else:
            name=f'82_c{start:05d}_m{ARMS.index(arm)}_{arm}_{start:05d}-{end:05d}.sh'
            out=SCRATCH/'out_qwen/fresh'
            command=[PYTHON,'scripts/strengthen_qwen.py','generate','--arm',arm,'--prompts',str(prompts),
                     '--start',str(start),'--end',str(end),'--out',str(out)]
            method=arm
            n=end-start
        path=staging/name
        body=env+[f'sha256sum --status -c {shlex.quote(str(checksum))}',
                  f'test -s {shlex.quote(str(NOTES / "qwen_download_receipt.json"))}',shlex.join(command)]
        with path.open('x') as stream:
            stream.write('\n'.join(body)+'\n')
        subprocess.run(['bash','-n',str(path)],check=True)
        image_out=out if block=='qwen_smoke' else out/arm/'4096x4096'
        tasks.append({'name':name,'method':method,'seed':0,'start':start,'end':end,'n':n,
                      'prompt_ids':[row['id'] for row in rows[start:end]],'out':str(image_out),
                      'staged':str(path),'sha256':file_sha256(path),'command':command})
    value={'timestamp':datetime.now(timezone.utc).isoformat(),'block':block,'tasks':tasks,
           'n_tasks':len(tasks),'n_generate':sum(task['n'] for task in tasks),'n_reuse':0,
           'prompts':str(prompts),'prompts_sha256':file_sha256(prompts),'checksum_file':str(checksum),
           'sources':{str(p):file_sha256(p) for p in sources},
           'prereg_sha256_at_queue':file_sha256(ROOT/'.agent/notes/prereg_strengthen.md'),
           'model_revision':MODEL_REVISION,'status':'staged only, publication is a separate action'}
    with manifest.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'block':block,'tasks':len(tasks),'images':value['n_generate'],'manifest':str(manifest)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('block',choices=('qwen_smoke','qwen'))
    prepare(p.parse_args().block)
